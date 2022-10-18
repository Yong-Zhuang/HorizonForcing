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
    samples = np.zeros((0,window_size,dimension))
    for i in np.arange(0,sequence.shape[0],stride):
        # find the end of this window
        end_ix = i + window_size
        if np.isnan(sequence[i:end_ix]).any():  
            continue
        print(f"generate sample start at {i}")    
		# check if we are beyond the sequence
        if end_ix > sequence.shape[0]:
            break
        #print(f"sequence[i:end_ix] {sequence[i:end_ix]}")   
        samples = np.append(samples,sequence[i:end_ix].reshape((1,window_size,dimension)),axis=0)  
    return samples

parser = argparse.ArgumentParser(
    description="Data Generator for Horizon Forcing Experiments."
)
parser.add_argument(
    "-s",
    "--system",
    type=str.lower,
    choices=["lorenz", "accelerometer", "gait_force", "roaming_worm", "electricity"],
    default="lorenz",
    help="System to build data for."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Collect arguments

    setting = config.EXP_SETTING[args.system]
    dataset_name = config.get_dataset_name(args.system)
    # If the output directory doesn't 
    # exist, try to create it.
    path = f"{config.DATA_FOLDER}/{dataset_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    if "lorenz" in dataset_name:
        lorenz = chaos_sys.Lorenz(setting["delta-t"], setting["total_steps"])
        sequence =lorenz.single_sequence_generating()[setting["burn_steps"]:]
    else:
        sequence = pd.read_csv(f"data/datasets/{setting["file"]}").values
    samples = split_sequence(sequence, setting["window"],setting["stride"])
    train_samples, test_samples = samples[:setting["n_training"]], samples[setting["n_training"]:]
        
    # Names of data files.
    train_output_file = f"{path}/train.npy"

    test_output_file = f"{path}/test.npy"
    # Save the first n_examples as train,
    # rest as test.
    np.save(train_output_file, train_samples)
    np.save(test_output_file, test_samples)
    print(f"training samples: {train_samples.shape}; test samples: {test_samples.shape}")