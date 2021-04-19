import argparse
import glob
import json
import os
import pdb
from os.path import dirname

import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform as transform
import tensorflow as tf
import tqdm
import ubelt as ub

import utils
from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
from PolygonModel import PolygonModel
from utils import save_to_json
from vis_predictions import main as vis

#
tf.logging.set_verbosity(tf.logging.INFO)
# --
flags = tf.flags
FLAGS = flags.FLAGS
# ---
flags.DEFINE_string(
    "PolyRNN_metagraph",
    "models/poly/polygonplusplus.ckpt.meta",
    "PolygonRNN++ MetaGraph ",
)
flags.DEFINE_string(
    "PolyRNN_checkpoint", "models/poly/polygonplusplus.ckpt", "PolygonRNN++ checkpoint "
)
flags.DEFINE_string(
    "EvalNet_checkpoint", "models/evalnet/evalnet.ckpt", "Evaluator checkpoint "
)
flags.DEFINE_string(
    "GGNN_metagraph", "models/ggnn/ggnn.ckpt.meta", "GGNN poly MetaGraph "
)
flags.DEFINE_string("GGNN_checkpoint", "models/ggnn/ggnn.ckpt", "GGNN poly checkpoint ")
flags.DEFINE_string("InputFolder", "imgs/", "Folder with input image crops")
flags.DEFINE_string("OutputFolder", "output/", "OutputFolder")
flags.DEFINE_boolean("Use_ggnn", True, "Use GGNN to postprocess output")

_BATCH_SIZE = 1
_FIRST_TOP_K = 5


def rect_to_box(xyxy):
    lx, ty, rx, by = xyxy
    center = (np.mean([lx, rx]), np.mean([ty, by]))
    longer_half = max(rx - lx, by - ty) / 2.0
    square_xyxy = [
        int(round(x))
        for x in [
            center[0] - longer_half,
            center[1] - longer_half,
            center[0] + longer_half,
            center[1] + longer_half,
        ]
    ]
    return square_xyxy


class PolygonRefiner:
    def __init__(self, infer=False):
        self.run_inference = infer

        self.last_image = None
        self.last_image_name = None
        self.index = 0
        if not self.run_inference:
            return

        # Model setup
        self.evalGraph = tf.Graph()
        self.polyGraph = tf.Graph()
        # Evaluator Network
        tf.logging.info("Building EvalNet...")
        with self.evalGraph.as_default():
            with tf.variable_scope("discriminator_network"):
                self.evaluator = EvalNet(_BATCH_SIZE)
                self.evaluator.build_graph()
            self.saver = tf.train.Saver()

            # Start session
            self.evalSess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True), graph=self.evalGraph
            )
            self.saver.restore(self.evalSess, FLAGS.EvalNet_checkpoint)

        # PolygonRNN++
        tf.logging.info("Building PolygonRNN++ ...")
        self.model = PolygonModel(FLAGS.PolyRNN_metagraph, self.polyGraph)

        self.model.register_eval_fn(
            lambda input_: self.evaluator.do_test(self.evalSess, input_)
        )

        self.polySess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True), graph=self.polyGraph
        )

        self.model.saver.restore(self.polySess, FLAGS.PolyRNN_checkpoint)

        if FLAGS.Use_ggnn:
            self.ggnnGraph = tf.Graph()
            tf.logging.info("Building GGNN ...")
            self.ggnnModel = GGNNPolygonModel(FLAGS.GGNN_metagraph, self.ggnnGraph)
            self.ggnnSess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True), graph=self.ggnnGraph
            )

            self.ggnnModel.saver.restore(self.ggnnSess, FLAGS.GGNN_checkpoint)

    def infer(self, image_np, crop_path, output_folder):
        # TODO see if we can get some batch parellism
        image_np = np.expand_dims(image_np, axis=0)
        preds = [
            self.model.do_test(self.polySess, image_np, top_k)
            for top_k in range(_FIRST_TOP_K)
        ]

        # sort predictions based on the eval score and pick the best
        preds = sorted(preds, key=lambda x: x["scores"][0], reverse=True)[0]

        if FLAGS.Use_ggnn:
            polys = np.copy(preds["polys"][0])
            feature_indexs, poly, mask = utils.preprocess_ggnn_input(polys)
            preds_gnn = self.ggnnModel.do_test(
                self.ggnnSess, image_np, feature_indexs, poly, mask
            )
            output = {"polys": preds["polys"], "polys_ggnn": preds_gnn["polys_ggnn"]}
        else:
            output = {"polys": preds["polys"]}

        # dumping to json files
        save_to_json(output_folder, crop_path, output)

    def refine(self, image_file, corners, output_dir):
        if self.last_image_name == image_file:
            image_np = self.last_image
        else:
            image_np = io.imread(image_file)
            self.last_image_name = image_file
            self.last_image = image_np.copy()
        # Creating the graphs
        lx, ty, rx, by = rect_to_box(corners)
        image_np = image_np[ty:by, lx:rx]
        if image_np.size == 0:
            return
        image_np = transform.resize(image_np, (224, 224))
        output_file = os.path.join(
            output_dir,
            "{}_{:06d}.png".format(
                os.path.basename(image_file.replace(".", "_")), self.index
            ),
        )
        # TODO Consider saving to a consistent temp file
        io.imsave(output_file, image_np)
        if self.run_inference:
            self.infer(image_np, output_file, output_dir)
        self.index += 1

    def process_file(self, annotation_file, image_basename, chip_dir, output_dir):
        ub.ensuredir(chip_dir, mode=0o0777, recreate=True)
        ub.ensuredir(output_dir, mode=0o0777, recreate=True)

        df = pd.read_csv(annotation_file, names=range(16), skiprows=2)
        corners = df.iloc[:, 3:7]
        filenames = df.iloc[:, 1]
        for (corner, filename) in tqdm.tqdm(zip(corners.iterrows(), filenames)):
            self.refine(os.path.join(image_basename, filename), corner[1], output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--image-dir")
    parser.add_argument("--chip-dir", default="imgs")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--show-ggnn", action="store_true", help="visualize ggnn")
    parser.add_argument(
        "--infer", action="store_true", help="actually perform inference on the chips"
    )
    args = parser.parse_args()
    if args.image_dir is None:
        args.image_dir = dirname(args.annotation_file)

    return args


if __name__ == "__main__":
    args = parse_args()
    refiner = PolygonRefiner(infer=args.infer)
    refiner.process_file(
        args.annotation_file, args.image_dir, args.chip_dir, args.output_dir
    )
    vis(args.output_dir, args.show_ggnn)
