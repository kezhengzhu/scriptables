# Plotting output data .xvg files in a meaningful way
# Using matplotlib and creating png files

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math


class Graph(object):
    def __init__(self, xlabels=None, ylabels=None, plots=None, figsize=(12.8,6.4), subplots=1, titles=None):
        # List of xlabels and ylabels: should be no longer than number of subplots
        if xlabels == None: xlabels = []
        if ylabels == None: ylabels = []
        if titles == None: titles = []
        if plots == None: plots = []
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.titles = titles
        if len(self.xlabels) > subplots:
            self.xlabels = self.xlabels[:subplots]
        if len(self.ylabels) > subplots:
            self.ylabels = self.ylabels[:subplots]
        if len(self.titles) > subplots:
            self.titles = self.titles[:subplots]

        # Setting up the figure environment and calculating number of rows and cols required
        # Always increase columns then rows, always aim for square (n x n)
        # (1x1), (1x2), (2x2), (2x3), (3x3) etc etc
        self.fig = plt.figure(figsize=figsize)
        self.subplots = subplots
        ncols = math.ceil(math.sqrt(self.subplots))
        self.subplot_shape = (math.ceil(self.subplots/ncols), ncols)

        # Setting up figure drawing
        self.set_subplots()

        # Dictionary of plots, each plot is a key with assign value 
        # as the subplot at which they are plotted
        # In init, we will just assume all plots go into subplot 1
        self.plots = {}

        for pl in plots:
            self.plots[pl] = 1

    def add_plots(self, *plots, subplot=1):
        # Check if the no. of arguments are correctly input and subplot index is correct
        if subplot > self.subplots or subplot <= 0:
            raise Exception("Subplot index out of range, Graph object only has {:d} subplots.".format(self.subplots))
        for pl in plots:
            if not isinstance(pl, Plot):
                print("Skipping plot as argument instance is not Plot object.")
            else:
                self.plots[pl] = subplot

    def set_xlabels(self, *args, add=False):
        # Function changes xlabel or add onto xlabel
        if add:
            for arg in args:
                assert(isinstance(arg, str))
                self.xlabels.append(arg)
            self.xlabels = self.xlabels[:self.subplots]
        else:
            self.xlabels = []
            for i in range(len(args)):
                if i < self.subplots:
                    assert(isinstance(args[i], str))
                    self.xlabels.append(args[i])

    def set_titles(self, *args, add=False):
        # Function changes titles or add onto titles
        if add:
            for arg in args:
                assert(isinstance(arg, str))
                self.titles.append(arg)
            self.titles = self.titles[:self.subplots]
        else:
            self.titles = []
            for i in range(len(args)):
                if i < self.subplots:
                    assert(isinstance(args[i], str))
                    self.titles.append(args[i])

    def set_ylabels(self, *args, add=False):
        # Function changes ylabel or add onto ylabel
        if add:
            for arg in args:
                assert(isinstance(arg, str))
                self.ylabels.append(arg)
            self.ylabels = self.ylabels[:self.subplots]
        else:
            self.ylabels = []
            for i in range(len(args)):
                if i < self.subplots:
                    assert(isinstance(args[i], str))
                    self.ylabels.append(args[i])

    def change_subplots(self, subplots):
        # Changing the no. of subplots
        self.subplots = subplots
        ncols = math.ceil(math.sqrt(self.subplots))
        self.subplot_shape = (math.ceil(self.subplots/ncols), ncols)
        self.set_subplots()

    def set_subplots(self):
        # Function clears figure and set up the subplots onto the figure
        self.fig.clf()
        self.subplot = []
        for subp in range(self.subplots):
            ax = self.fig.add_subplot(*self.subplot_shape,subp+1)
            ax.margins(x=0)
            self.subplot.append(ax)

        return self.subplot

    def set_labels(self):
        # Set titles and labels onto subplots
        for i in range(self.subplots):
            if i >= len(self.titles):
                self.subplot[i].set_title("Title {}".format(i+1))
            else:
                self.subplot[i].set_title(self.titles[i])

            if i >= len(self.xlabels):
                self.subplot[i].set_xlabel("X-axis {}".format(i+1))
            else:
                self.subplot[i].set_xlabel(self.xlabels[i])

            if i >= len(self.ylabels):
                self.subplot[i].set_ylabel("Y-axis {}".format(i+1))
            else:
                self.subplot[i].set_ylabel(self.ylabels[i])

    def plot_graphs(self):
        # Function draws all the plots that have been loaded onto the respective subplots
        for pl in self.plots:
            sp = self.plots[pl]
            self.subplot[sp-1].plot(pl.x, pl.y, c=pl.color, label=pl.label)

    def draw(self, savefig=""):
        # Draw function should be the one called. Basically uses existing functions to draw
        self.set_subplots()
        self.set_labels()
        self.plot_graphs()
        if len(savefig) > 0:
            self.fig.savefig(savefig)
        plt.show()


class Plot(object):
    def __init__(self, x=None, y=None, label="plot", xvgfile="", color="r"):
        if x == None: x = []
        if y == None: y = []
        self.x = x
        self.y = y
        self.label = label
        self.color = color

        if len(xvgfile) > 4 and xvgfile[-4:].lower() == ".xvg":
            self.read_xvg(xvgfile)

    def read_xvg(self, filename):
        if (not isinstance(filename, str)) or filename[-4:] != ".xvg":
            raise Exception("read_xvg failed due to wrong input or wrong file extension")

        with open(filename) as f:
            for line in f:
                if line[0] == "#" or line[0] == "@":
                    pass
                else:
                    cols = line.split()
                    if len(cols) == 2:
                        self.x.append(float(cols[0]))
                        self.y.append(float(cols[1]))

    def clear_int(self, interval):
        x = []
        y = []
        for i in range(len(self.x)):
            if i%interval == 0:
                x.append(self.x[i])
                y.append(self.y[i])
            elif i == len(self.x)-1:
                x.append(self.x[i])
                y.append(self.y[i])
        self.x = x
        self.y = y

    def shorten(self, ratio):
        x = []
        y = []
        assert(ratio > 0 and ratio < 1)
        length = int(len(self.x) * ratio)
        for i in range(length):
            x.append(self.x[i])
            y.append(self.y[i])
        self.x = x
        self.y = y 

def main():
    fn1 = "pressure_nvteq.xvg"
    fn2 = "pressure_nvt.xvg"
    p1 = Plot(xvgfile=fn1)
    p12 = Plot(xvgfile="pressure_nvteq2.xvg", color="b")
    p2 = Plot(xvgfile=fn2)
    p22 = Plot(xvgfile="pressure_nvt2.xvg", color="b")

    print(len(p1.x), len(p2.x))

    g = Graph(subplots=2)
    g.add_plots(p1, p12, subplot=1)
    g.add_plots(p2, p22, subplot=2)
    g.set_xlabels("Time (ps)", "Time (ps)")
    g.set_ylabels("Pressure (bar)","Pressure (bar)")
    g.set_titles(*["Pressure v Time"]*2)
    g.draw()


if __name__ == '__main__':
    main()