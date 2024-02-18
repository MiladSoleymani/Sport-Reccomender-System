import sys
import os

sys.path.append(os.getcwd())

from typing import Dict

from data.dataset import ATHLXDataset
from model.multi_class_classification import NeuralNet

import shutil
import argparse

from keras.models import load_weights


def inference(conf: Dict) -> None:
    dataset = ATHLXDataset(
        parameters_data_path=conf["parameters_data_path"],
        labels_data_path=conf["labels_data_path"],
        num_top_spotrts=conf["num_top_spotrts"],
    )
    print(dataset)

    x, y = dataset.__getitem__()

    # load weights
    model = load_weights(conf["weight_path"])
    # summarize model.
    model.summary()

    score = model.evaluate(x, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameters_data_path",
        type=str,
        default="./dataset/parameters",
    )
    parser.add_argument(
        "--labels_data_path",
        type=str,
        default="./dataset/labels",
    )

    parser.add_argument(
        "--num_top_spotrts",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        default="./results/checkpoint.weights.h5",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)

    # transforms = Compose([ToNumpy()])

    inference(conf)
