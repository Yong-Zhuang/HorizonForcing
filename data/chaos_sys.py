"""
 - To build Lorenz '63 system data and Rossler system data as in paper, run with default
arguments (will save to ./data/ as default):

python generate_dataset.py --output_dir="/path/to/data"

 - To build Rossler system data as in paper, run as:

python generate_dataset.py --system="rossler" --output_dir="/path/to/data"
"""

import os
import numpy as np
import argparse
from abc import abstractmethod
from pathlib import Path
# We use multiprocessing to build example trajectories
# in parallel
from multiprocessing import Pool
from multiprocessing import sharedctypes
# Use scipy for initial value problem
# solving.
from scipy.integrate import solve_ivp, odeint



# ----------------------------------------------------------------
class ChaosBase:
    def __init__(self, del_t, n_steps):  
        """        
        Args:
            del_t: Time step magnitude, delta-t.. 
            n_steps: Number of steps per sequence.
        """     
        self.del_t = del_t
        self.n_steps = n_steps
        self.time_steps = np.arange(0, self.del_t * n_steps, self.del_t)
        self.derivative = self.get_derivative(*self.meta["params"])


    
    """     
    Majority of work done here. This is the base
    function that is parallelized; each trip through
    here creates one data sample and saves it to 
    the correct memory in the shared array.
    We use the default fourth-order Runge-Kutta 
    method to solve the IVPs and generate trajectories
    """     
    def build_data_series(self, shared_array, series_index, atol=1e-8, rtol=1e-8):
        """        
        Args:
            atol: Absolute tolerance for IVP solver.
            rtol: Relative Tolerance for IVP solver.
        """     
        x_high, x_low = self.meta["x_high"], self.meta["x_low"]
        y_high, y_low = self.meta["y_high"], self.meta["y_low"]
        z_high, z_low = self.meta["z_high"], self.meta["z_low"]

        X_ref = np.ctypeslib.as_array(shared_array)

        series = X_ref[series_index]

        x0 = np.random.rand(3)
        x0[0] = x0[0] * (x_high - x_low) + x_low
        x0[1] = x0[1] * (y_high - y_low) + y_low
        x0[2] = x0[2] * (z_high - z_low) + z_low

        series = solve_ivp(
            fun=self.derivative, 
            t_span=(self.time_steps[0], self.time_steps[-1]),
            y0=x0, 
            t_eval=self.time_steps,
            rtol=rtol, atol=atol,
            dense_output=True
        ).y

        X_ref[series_index] = np.copy(series.T)
    
    def multi_sequence_generating(self, n_samples, n_steps, b_steps):
        dimension = self.meta["dimension"]

        # Allocate memory for datasets.
        final_array= np.ctypeslib.as_ctypes(  
            np.zeros((n_samples, n_steps, dimension))
        )

        shared_array= sharedctypes.RawArray(
            final_array._type_, final_array
        )

        series_indices = np.arange(n_samples)

        # Build one example trajectory for each integer
        # in [0, total_n). Store in the shared memory at
        # that position.
        p = Pool()
        res = p.starmap(self.build_data_series, [(shared_array, series_indices)])

        # Final array is our total (train + test)
        # dataset.
        final_array = np.ctypeslib.as_array(shared_array)
        return final_array


    def single_sequence_generating(self):
         # -s lorenz_s -ns 58000 -bs 8000 -dt 0.01 -n 8500   ; 1201
        init_sate = [1,1,1]
        sequence = odeint(self.derivative(*self.meta["params"]), init_sate, self.time_steps, tfirst = True)
        return sequence

    @abstractmethod
    def get_derivative(sharedctypes): 
        pass

class Lorenz(ChaosBase): 
    def __init__(self, del_t, n_steps, atol, rtol):  
        # self.metadata dictionaries to store generation parameters
        # used for each attractor type.
        self.meta = {
            "x_low": -20,
            "x_high": 20,
            "y_low": -20,
            "y_high": 20,
            "z_low":0,
            "z_high":50,
            "dimension":3,
            "params": [28.0, 10.0, 8.0 / 3.0]
        }       
        super(Lorenz, self).__init__(del_t, n_steps, atol, rtol)    


    # Use closures to return self.derivative functions
    # with the given parameters for each system.
    def get_derivative(self, rho, sigma, beta):
        def f(t, state):
                x, y, z = state  # Unpack the state vector
                return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives
        self.derivate = f

class Rossler(ChaosBase):  
    def __init__(self, del_t, n_steps, atol, rtol):     
        self.meta = {
            "x_low": -15,
            "x_high": 15,
            "y_low": -15,
            "y_high": 15,
            "z_low":0,
            "z_high":50,
            "dimension":3,
            "params": [0.1, 0.1, 18.0]
        }  
        super(Rossler, self).__init__(del_t, n_steps, atol, rtol)  

    def get_derivative(self,a,b,c):
        def f(t, state):
            x, y, z = state  # Unpack the state vector
            return -1 * y - z, x + (a * y), b + (z * (x - c))
        self.derivate = f 
