"""
Build simulated data sets for Horizon Forcing experiments.

Usage:

To build Lorenz '63 system data as in paper, run with default
arguments (will save to ./data/ as default):

python generate_dataset.py --output_dir="/path/to/data"

To build Rossler system data as in paper, run as:

python generate_dataset.py --system="rossler" --output_dir="/path/to/data"
"""
import argparse
import numpy as np
import os
import time

# We use multiprocessing to build example trajectories
# in parallel
from multiprocessing import Pool
from multiprocessing import sharedctypes

from pathlib import Path

# Use scipy for initial value problem
# solving.
from scipy.integrate import solve_ivp

# Use closures to return derivative functions
# with the given parameters for each system.
def get_lorenz_derivative(rho, sigma, beta):
    def f(t, state):
            x, y, z = state  # Unpack the state vector
            return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    return f

def get_rossler_derivative(a,b,c):
    def f(t, state):
        x, y, z = state  # Unpack the state vector
        return -1 * y - z, x + (a * y), b + (z * (x - c))

    return f

# Majority of work done here. This is the base
# function that is parallelized; each trip through
# here creates one data sample and saves it to 
# the correct memory in the shared array.
#
# We use the default fourth-order Runge-Kutta 
# method to solve the IVPs and generate trajectories
def build_data_series(
    series_index
):
    global _DERIVATIVE
    global _SHARED_ARRAY
    global _TIME_STEPS

    X_ref = np.ctypeslib.as_array(_SHARED_ARRAY)

    series = X_ref[series_index]
    derivative = _DERIVATIVE
    time_steps = _TIME_STEPS

    x0 = np.random.rand(3)
    x0[0] = x0[0] * (x_high - x_low) + x_low
    x0[1] = x0[1] * (y_high - y_low) + y_low
    x0[2] = x0[2] * (z_high - z_low) + z_low

    series = solve_ivp(
        fun=derivative, 
        t_span=(time_steps[0], time_steps[-1]),
        y0=x0, 
        t_eval=time_steps,
        rtol=rtol, atol=atol,
        dense_output=True
    ).y

    X_ref[series_index] = np.copy(series.T)

# ----------------------------------------------------------------
# Metadata dictionaries to store generation parameters
# used for each attractor type.

_LORENZ_META = {
    "x_low": -20,
    "x_high": 20,
    "y_low": -20,
    "y_high": 20,
    "z_low":0,
    "z_high":50,
    "dimension":3,
    "params" = [28.0, 10.0, 8.0 / 3.0]
}
_ROSSLER_META = {
    "x_low": -15,
    "x_high": 15,
    "y_low": -15,
    "y_high": 15,
    "z_low":0,
    "z_high":50,
    "dimension":3,
    "params" = [0.1, 0.1, 18.0]
}

# ----------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Data Generator for Horizon Forcing Experiments."
)
parser.add_argument(
    "-s",
    "--system",
    type=str.lower,
    choices=["lorenz","rossler"],
    default="lorenz",
    help="System to build data for."
)
parser.add_argument(
    "-n",
    type=int,
    default=15000,
    help="Number of training examples in dataset."
    )
parser.add_argument(
    "-n_test",
    type=int,
    default=500,
    help="Number of testing examples in dataset."
    )
parser.add_argument(
    "-ns",
    type=int,
    default=1000,
    help="Number of steps per example."
    )
parser.add_argument(
    "-dt",
    type=float,
    default=0.05,
    help="Time step magnitude, delta-t."
    )
parser.add_argument(
    "-o",
    "--output_dir",
    default="./data/",
    help="Path where data should be saved."
)
if __name__ == "__main__":
    args = parser.parse_args()

    # Collect arguments
    n_examples = args.n
    n_test_examples = args.n_test
    total_n = n_examples + n_test_examples

    n_steps = args.ns
    dataset_name = args.s
    del_t = args.dt
    output_dir = args.output_dir

    # If the output directory doesn't 
    # exist, try to create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the metadata file based on 
    # argument.
    if dataset_name == "lorenz":
        meta = _LORENZ_META
        _DERIVATIVE = get_lorenz_derivative(
            *meta.params
        )
    else:
        meta = _ROSSLER_META
        _DERIVATIVE = get_rossler_derivative(
            *meta.params
        )

    dimension = meta["dimension"]

    # Allocate memory for datasets.
    _FINAL_ARRAY = np.ctypeslib.as_ctypes(
        np.zeros((total_n, n_steps, dimension))
    )

    _SHARED_ARRAY = sharedctypes.RawArray(
        _FINAL_ARRAY._type_, _FINAL_ARRAY
    )

    _TIME_STEPS = np.arange(0, del_t * n_steps, del_t)

    series_indices = np.arange(total_n)

    # Build one example trajectory for each integer
    # in [0, total_n). Store in the shared memory at
    # that position.
    p = Pool()
    res = p.map(build_data_series, series_indices)

    # Final array is our total (train + test)
    # dataset.
    final_array = np.ctypeslib.as_array(_SHARED_ARRAY)

    # Names of data files.
    train_output_file = Path(output_dir)/(
        f"X_train_{dataset_name}.npy"
    )

    test_output_file = Path(output_dir)(
        f"X_test_{dataset_name}.npy"
    )

    # Save the first n_examples as train,
    # rest as test.
    np.save(train_output_file, final_array[:n_examples])
    np.save(test_output_file, final_array[n_examples:])