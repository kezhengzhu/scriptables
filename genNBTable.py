
# This is a program that auto generates table and non-bonded forcefield .itp
# files. This can also be loaded as a module for the following functions:
# writeffnb(): creates non-bonded forcefield itp file
# genTable(): creates forcefield interaction tables

import numpy as np
import pandas as pd 

print("Loading functions from genNBTable.py")

def writeffnb():
    fout = open("ffnonbonded.itp","w")
    fout.close()

def genTable():


def main():
    print("Hello World!")
    return

if __name__ == '__main__':
    main()