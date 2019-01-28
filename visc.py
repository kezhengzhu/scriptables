import numpy as np
import pandas as pd 
import sys
from plotxvg import *

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
    checkerr(isinstance(lags, int),
        "No. of lags must be given as an int")

    if y.ndim == 1:
        nrows = y.size
        y = np.reshape(y, (nrows, 1))
    (nrows, ncols) = y.shape
    checkerr(lags < nrows, 
        "No. of lags must be less than length of vector(s)")
    # Calculations
    ybar = np.mean(y,0) # array size = ncols
    N = nrows
    result = np.zeros((nrows, ncols))
    yvar = np.matmul((y - ybar).transpose(), (y - ybar))
    print("Begin ACF calculations:")
    for col in range(ncols):
        for lag in range(lags+1):
            print("\rDataset column: {:3d}, at lag no. = {:8d}".format(col,lag), end=" ")
            cross_sum = 0
            for i in range(nrows-lag):
                cross_sum += (y[i, col] - ybar[col]) * (y[i+lag, col] - ybar[col])
            result[lag,col] = cross_sum/yvar[col,col];
        print()

    return result

def otheracf(y):
    # Checking types and sizes
    # Only lists and np.ndarrays are accepted
    checkerr(isinstance(y, np.ndarray) or isinstance(y, list),
        "ACF input only accepts np.ndarray and lists")
    if isinstance(y, list): y = np.array(y)
    checkerr(y.dtype != np.dtype('O'),
        "ndarray dtype wrong! Likely wrongly structured 2d lists")

    checkerr(y.ndim <= 2 and y.ndim >= 1,
        "ndarray too many dimensions!")
    checkerr(isinstance(lags, int),
        "No. of lags must be given as an int")

    if y.ndim == 1:
        nrows = y.size
        y = np.reshape(y, (nrows, 1))
    (nrows, ncols) = y.shape
    return

def s_visc(time, pressure, temp, boxlen): # boxlen units [nm]
    return


def main():
    return


if __name__ == '__main__':
    main()


