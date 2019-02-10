# Plotting output data .xvg files in a meaningful way
# Using matplotlib and creating png files

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math, random
import copy


class Graph(object):
    '''
    Class of Graph contains the following information:
    How many subplots on a pyplot figure, figure size, and axis labels for the
    different subplots. As well as a dictionary of plots (as key) corresponding
    to the subplot that they are plotted onto (as value)
    '''
    def __init__(self, xlabels=None, ylabels=None, plots=None, figsize=(12.8,6.4), subplots=1, titles=None, legends=False):
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

        self.spxlim = [None] * self.subplots
        self.spylim = [None] * self.subplots

        # Setting up figure drawing
        self.set_subplots()

        # Dictionary of plots, each plot is a key with assign value 
        # as the subplot at which they are plotted
        # In init, we will just assume all plots go into subplot 1
        self.plots = {}

        for pl in plots:
            self.plots[pl] = 1

        self.legends = legends

    def __repr__(self):
        s = "Graph of {:d} subplots: \n".format(self.subplots)
        for i in range(self.subplots):
            n = sum(pl == i+1 for pl in self.plots.values())
            s += "\tSubplot {:d}: {:d} plots".format(i+1, n)
            if i != self.subplots - 1:
                s += "\n"
        return s


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
        self.spxlim = [None] * self.subplots
        self.spylim = [None] * self.subplots
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

    def ylim(self, bottom=None, top=None, subplot=1, y_range=None):
        (b,t) = self.subplot[subplot-1].get_ylim()
        if not isinstance(y_range, tuple):
            if isinstance(top, float) or isinstance(top, int):
                t = top
            if isinstance(bottom, float) or isinstance(bottom, int):
                b = bottom
        elif isinstance(y_range, tuple) and len(y_range) == 2:
            (b,t) = y_range
        self.spylim[subplot-1] = (b,t)

    def xlim(self, left=None, right=None, subplot=1, x_range=None):
        (l,r) = self.subplot[subplot-1].get_xlim()
        if not isinstance(x_range, tuple):
            if isinstance(left, float) or isinstance(left, int):
                l = left
            if isinstance(right, float) or isinstance(right, int):
                r = right
        elif isinstance(x_range, tuple) and len(x_range) == 2:
            (l,r) = x_range
        self.spxlim[subplot-1] = (l,r)

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

    def set_lims(self):
        # Set the limits onto the subplots
        for i in range(self.subplots):
            if isinstance(self.spxlim[i], tuple):
                self.subplot[i].set_xlim(*self.spxlim[i])
            if isinstance(self.spylim[i], tuple):
                self.subplot[i].set_ylim(*self.spylim[i])

    def plot_graphs(self):
        # Function draws all the plots that have been loaded onto the respective subplots
        for pl in self.plots:
            sp = self.plots[pl]
            if pl.axes == 'loglog':
                self.subplot[sp-1].loglog(pl.x, pl.y, color=pl.color, marker=pl.marker, linestyle=pl.style, label=pl.label)
            elif pl.axes == 'semilogx':
                self.subplot[sp-1].semilogx(pl.x, pl.y, color=pl.color, marker=pl.marker, linestyle=pl.style, label=pl.label)
            elif pl.axes == 'semilogy':
                self.subplot[sp-1].semilogy(pl.x, pl.y, color=pl.color, marker=pl.marker, linestyle=pl.style, label=pl.label)
            else:
                self.subplot[sp-1].plot(pl.x, pl.y, color=pl.color, marker=pl.marker, linestyle=pl.style, label=pl.label)

    def draw(self, savefig="", dpi=600):
        # Draw function should be the one called. Basically uses existing functions to draw
        self.set_subplots()
        self.set_labels()
        self.plot_graphs()
        self.set_lims()
        if self.legends:
            for sp in self.subplot:
                sp.legend()
        if isinstance(savefig,str) and len(savefig) > 0:
            self.fig.savefig(savefig, bbox_inches="tight", dpi=dpi)
        plt.show()

    def add_hline(self, yval, subplot=1, color='k', style='--'):
        x_range = self.spxlim[subplot-1]
        if not isinstance(x_range, tuple):
            x_range = self.subplot[subplot-1].get_xlim()
            self.spxlim[subplot-1] = x_range

        pl = Plot(list(x_range),[yval, yval], label="y ={:4.1f}".format(yval), color=color, style=style)
        self.add_plots(pl, subplot=subplot)

    def add_vline(self, xval, subplot=1, color='k', style='--'):
        y_range = self.spylim[subplot-1]
        if not isinstance(y_range, tuple):
            y_range = self.subplot[subplot-1].get_ylim()
            self.spylim[subplot-1] = y_range
            
        pl = Plot([xval, xval], list(y_range), label="x ={:4.1f}".format(xval), color=color, style=style)
        self.add_plots(pl, subplot=subplot)


class Plot(object):
    colors = list('rbgcmyk')
    markers = list('.,ov^<>1234sp*hHxDd|_')
    styles = ['-', '--', '-.', ':']
    '''
    Class of Plot:
    Contains x and y values, as well as labels and file origin. Also contains
    color information and plot style (TODO)
    '''
    def __init__(self, x=None, y=None, label="plot", xvgfile="", color="", marker="", style="-", axes="linear"):
        # Initialises plots by taking in ndarray or list as points, else initialise empty plot
        # if xvgfile is valid takes in xvgfile and appends to plot. 
        if (not (isinstance(x, list) or isinstance(x, np.ndarray))): x = []
        if (not (isinstance(y, list) or isinstance(y, np.ndarray))): y = []
        self.x = x
        self.y = y
        self.label = label
        if color in Plot.colors:
            self.color = color
        else:
            ind = math.floor(random.random()*7)
            self.color = Plot.colors[ind]
        self.marker = marker
        self.style = style
        self.axes = axes.lower() # 'linear', 'semilogx', 'semilogy', 'loglog'

        if len(xvgfile) > 4 and xvgfile[-4:].lower() == ".xvg":
            # There is no clearing function here also xvg files APPENDS to existing 
            # x and y list / array
            self.read_xvg(xvgfile)

    def __repr__(self):
        return "Plot object <{}>".format(self.label)

    def rm_pointers(self):
        # Replaces x and y lists with a copy of it to avoid pointer issues
        if isinstance(self.x, list):
            self.x = copy.copy(self.x)
        if isinstance(self.y, list):
            self.y = copy.copy(self.y)

    def tolist(self):
        # Converts ndarray x and y to lists
        if isinstance(self.x, np.ndarray):
            self.x = self.x.tolist()
        if isinstance(self.y, np.ndarray):
            self.y = self.y.tolist()

    def toarray(self):
        # Converts lists x and y to np.ndarrays
        if isinstance(self.x, list):
            self.x = np.array(self.x)
        if isinstance(self.y, list):
            self.y = np.array(self.y)

    def read_xvg(self, filename):
        # Reads an xvg file and tries to append it to x and y. 
        # Ensures that it works on both lists and ndarrays
        if (not isinstance(filename, str)) or filename[-4:] != ".xvg":
            raise Exception("read_xvg failed due to wrong input or wrong file extension")

        with open(filename) as f:
            for line in f:
                if line[0] == "#" or line[0] == "@":
                    pass
                else:
                    cols = line.split()
                    if len(cols) == 2:
                        if isinstance(self.x, list):
                            self.x.append(float(cols[0]))
                        elif isinstance(self.x, np.ndarray):
                            np.append(self.x, float(cols[0]))
                        if isinstance(self.y, list):
                            self.y.append(float(cols[1]))
                        elif isinstance(self.y, np.ndarray):
                            np.append(self.y, float(cols[1]))

    def clear_int(self, interval):
        # Function removes points at a fixed interval, keeping first and last value
        # i.e. [1,2,3,4,5,6,7,8,9,...98,99] removes at interval 4 gives
        # [1,5,9,13...97,99]
        x = []
        y = []
        for i in range(len(self.x)):
            if i%interval == 0:
                x.append(self.x[i])
                y.append(self.y[i])
            elif i == len(self.x)-1:
                x.append(self.x[i])
                y.append(self.y[i])
        if isinstance(self.x, list):
            self.x = x
        elif isinstance(self.x, np.ndarray):
            self.x = np.array(x)
        else:
            print("clear_int() failed, .x not a list or ndarray")
        if isinstance(self.y, list):
            self.y = y
        elif isinstance(self.y, np.ndarray):
            self.y = np.array(y)
        else:
            print("clear_int() failed, .y not a list or ndarray")

    def shorten(self, ratio):
        # Shortens the plot by a certain ratio.
        # 0.4 shorten of 1-10 gives 1-4
        x = []
        y = []
        assert(ratio > 0 and ratio < 1)
        length = int(len(self.x) * ratio)
        for i in range(length):
            x.append(self.x[i])
            y.append(self.y[i])
        if isinstance(self.x, list):
            self.x = x
        elif isinstance(self.x, np.ndarray):
            self.x = np.array(x)
        else:
            print("shorten() failed, .x not a list or ndarray")
        if isinstance(self.y, list):
            self.y = y
        elif isinstance(self.y, np.ndarray):
            self.y = np.array(y)
        else:
            print("shorten() failed, .y not a list or ndarray")

    def ravg_plot(self, color="b"):
        # Returns another Plot object which plots the running average of this plot
        # Running average calculation is made faster by taking the previous point,
        # multiplying it by it's index, adding current point and dividing by index + 1
        # i.e. average of 1st 100 points (0-99) = 10
        # current point = 20. Current running avg = (10*100 + 20) / (101) = 10.099
        x = []
        y = []
        for i in range(len(self.y)):
            x.append(self.x[i])
            if i == 0:
                y.append(self.y[i])
            else:
                elem = ((y[i-1]*i) + self.y[i]) / (i+1)
                y.append(elem)
        return Plot(x,y,"Running average of "+self.label,color=color)

    def set_color(self, color):
        # Set self.color to what the user desire
        assert(isinstance(color,str))
        if color in Plot.colors:
            self.color = color

    def set_style(self, style):
        # Set self.style to what the user desire
        assert(isinstance(style,str))
        if style in Plot.styles:
            self.style = style

    def set_marker(self, marker):        
        # Set self.marker to what the user desire
        assert(isinstance(marker,str))
        if marker in Plot.markers:
            self.marker = marker


def main():
    pl1 = Plot(xvgfile="temp8.xvg",color="r")
    # pl1.shorten(0.1)
    pl1.clear_int(1000)
    pl1a = pl1.ravg_plot(color="b")
    g = Graph()
    g.add_plots(pl1, pl1a)
    g.set_xlabels("Time (ps)")
    g.set_ylabels("Temp (K)")
    g.set_titles("Temp v Time")
    g.draw()
    # fn1 = "pressure_nvteq.xvg"
    # fn2 = "pressure_nvt.xvg"
    # p1 = Plot(xvgfile=fn1)
    # p12 = Plot(xvgfile="pressure_nvteq2.xvg", color="b")
    # p2 = Plot(xvgfile=fn2)
    # p22 = Plot(xvgfile="pressure_nvt2.xvg", color="b")

    # print(len(p1.x), len(p2.x))

    # g = Graph(subplots=2)
    # g.add_plots(p1, p12, subplot=1)
    # g.add_plots(p2, p22, subplot=2)
    # g.set_xlabels("Time (ps)", "Time (ps)")
    # g.set_ylabels("Pressure (bar)","Pressure (bar)")
    # g.set_titles(*["Pressure v Time"]*2)
    # g.draw()


if __name__ == '__main__':
    main()