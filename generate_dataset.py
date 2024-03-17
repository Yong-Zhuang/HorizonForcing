"""
Build simulated data sets for Horizon Forcing experiments.

Usage:

To build Lorenz '63 system data as in paper, run with default
arguments:

python generate_dataset.py -s="lorenz"
"""

import argparse
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from data import chaos_sys
import config


def split_sequence(sequence, window_size, stride):
    dimension = sequence.shape[-1]
    samples = np.zeros((0, window_size, dimension))
    for i in np.arange(0, sequence.shape[0], stride):
        # find the end of this window
        end_ix = i + window_size
        if np.isnan(sequence[i:end_ix]).any():
            continue
        # print(f"generate sample start at {i}")
        # check if we are beyond the sequence
        if end_ix > sequence.shape[0]:
            break
        # print(f"sequence[i:end_ix] {sequence[i:end_ix]}")
        samples = np.append(
            samples, sequence[i:end_ix].reshape((1, window_size, dimension)), axis=0
        )
    return samples


parser = argparse.ArgumentParser(
    description="Data Generator for Horizon Forcing Experiments."
)
parser.add_argument(
    "-s",
    "--system",
    type=str.lower,
    choices=[
        "lorenz",
        "rossler",
        "accelerometer",
        "dwelling_worm",
        "ecg",
        "ecosystem",
        "electricity",
        "gait_force",
        "gait_marker_tracker",
        "geyser",
        "mouse",
        "pendulum",
        "roaming_worm",
    ],
    default="lorenz",
    help="System to build data for.",
    required=True,
)

parser.add_argument(
    "-sub",
    "--sub",
    type=str.lower,
    default="default",
    help="Which subject?",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Collect arguments

    setting = config.EXP_SETTING[args.system][args.sub]
    dataset_name = config.get_dataset_name(args.system, args.sub)
    # If the output directory doesn't
    # exist, try to create it.
    path = f"{config.DATA_FOLDER}/{dataset_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.system == "ecg" or args.system == "pendulum":
        sequence_train = pd.read_csv(f"data/datasets/{setting['file']['train']}").values
        sequence_test = pd.read_csv(f"data/datasets/{setting['file']['test']}").values
        if args.system == "pendulum":
            sequence_train = sequence_train.transpose()
            sequence_test = sequence_test.transpose()
        print(
            f"sequence_train shape {sequence_train.shape, sequence_train[:3]}, sequence_test shape {sequence_test.shape, sequence_test[:3]}"
        )
        train_samples = split_sequence(
            sequence_train, setting["window"], setting["stride"]
        )
        test_samples = split_sequence(
            sequence_test, setting["window"], setting["stride"]
        )
    else:
        if args.system == "lorenz":
            lorenz = chaos_sys.Lorenz(setting["delta-t"], setting["total_steps"])
            sequence = lorenz.single_sequence_generating()[setting["burn_steps"] :]
        elif args.system == "rossler":
            rossler = chaos_sys.Rossler(setting["delta-t"], setting["total_steps"])
            sequence = rossler.single_sequence_generating()[setting["burn_steps"] :]
        else:
            sequence = pd.read_csv(f"data/datasets/{setting['file']}").values
            if args.system == "ecosystem":
                sequence = sequence.transpose()
        print(f"sequence shape {sequence.shape, sequence[:3]}")
        samples = split_sequence(sequence, setting["window"], setting["stride"])
        train_samples, test_samples = (
            samples[: setting["n_training"]],
            samples[setting["n_training"] :],
        )

    # Names of data files.
    train_output_file = f"{path}/train.npy"

    test_output_file = f"{path}/test.npy"
    # Save the first n_examples as train,
    # rest as test.
    np.save(train_output_file, train_samples)
    np.save(test_output_file, test_samples)
    print(
        f"training samples: {train_samples.shape}; test samples: {test_samples.shape}"
    )
