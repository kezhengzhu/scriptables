import numpy as np
import pandas as pd 

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

def acf(y, lags):
    # Checking types and sizes
    # Only lists and np.ndarrays are accepted
    checkerr(isinstance(y, np.ndarray) or isinstance(y, list), 
        "ACF input only accepts np.ndarray and lists")
    if isinstance(y, list): y = np.array(y)
    checkerr(y.dtype != np.dtype('O'),
        "ndarray dtype wrong! Likely wrongly structured 2d lists")

    checkerr(y.ndim <= 2 and y.ndim >= 1, 
        "ndarray too many dimensions!")
    # Calculations
    ybar = np.mean(y)
    N = 

