import sys
import os

sys.path.append(os.getcwd())

from typing import Dict

from data.dataset import ATHLXDataset
from model.multi_class_classification import NeuralNet

import shutil
import argparse

import keras
import keras_tuner


def train(conf: Dict) -> None:
    dataset = ATHLXDataset(
        parameters_data_path=conf["parameters_data_path"],
        labels_data_path=conf["labels_data_path"],
        num_top_spotrts=conf["num_top_spotrts"],
    )
    print(dataset)

    x, y = dataset.__getitem__()

    model = NeuralNet(
        conf=None,
        n_inputs=x.shape[1],
        n_outputs=y.shape[1],
    )
    print(model)

    # Check if the folder exists
    if os.path.exists(conf["save_path"]):
        # Remove the folder and its contents
        shutil.rmtree(conf["save_path"])
        print(f"The folder at {conf['save_path']} has been removed.")
    else:
        print(f"The folder at {conf['save_path']} does not exist.")

    # Create the folder
    os.makedirs(conf["save_path"])

    tuner = keras_tuner.RandomSearch(
        model.create_model,
        objective="val_accuracy",
        directory=conf["save_path"],
        overwrite=True,
        project_name="ATHLX",
        max_trials=50,
    )

    print(tuner.search_space_summary())

    tuner.search(
        x,
        y,
        epochs=10,
        verbose=1,
        callbacks=[keras.callbacks.TensorBoard(conf["save_path"])],
        validation_split=0.1,
    )

    print(tuner.results_summary(1))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    checkpoint_filepath = os.path.join(conf["save_path"], "checkpoint.weights.h5")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(
        x,
        y,
        epochs=conf["num_epoch"],
        validation_split=0.1,
        callbacks=[model_checkpoint_callback],
    )

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))


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
        "--num_epoch",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)

    # transforms = Compose([ToNumpy()])

    train(conf)
