import numpy as np 
import pandas as pd 
import math
import os
import datetime
import time

# Set-up file systems

# Define classes: simulation, subsimulation, variables, changes etc

class Simulation(object):
    def __init__(self, folderName=""):
        if len(folderName) == 0:
            try:
                dt = datetime.datetime.now()
                folderName = "gropy_sim_{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                os.mkdir('./' + folderName)
            except:
                time.sleep(1)
                dt = datetime.datetime.now()
                folderName = "gropy_sim_{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                os.mkdir('./' + folderName)
        self.dir = os.getcwd() + "/" + folderName
        self.beads = dict()
        self.molecules = dict()
        self.runs = dict()

class CGBeads(object):


class CGMole(object):


class MDRun(object):


