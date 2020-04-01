"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

This file contains wrapper functions for creation of output functions

Functions:
plot_histogram:
plot_2D_line
plot_2D_scatter
"""

from matplotlib import pyplot as plt

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (16, 9)

def plot_histogram(
        data,
        bins,
        title,
        label_x,
        label_y,
        figsize,
        dpi,
        output_path
        ):
    """
    Abstracted plotting of histogram

    Args:
    data: list or np.array, the data to plot in the histogram
    bins: int, the number of bins in the histogram
    title: str, the title of the plot
    label_x: str, the x-axis label
    label_y: str, the y-axis label
    figsize: pair of ints, size of output plot
    dpi:int, dpi of output plot
    output_path: str or None, if None then show graph, otherwise save it
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.hist(data, bins=bins)
    plt.title(title, fontsize=24)
    plt.xlabel(label_x, fontsize=20)
    plt.ylabel(label_y, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Display or save
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_2D_scatter(
        data_x,
        data_y,
        title,
        label_x,
        label_y,
        figsize,
        dpi,
        output_path=None
        ):
    """
    Abstracted plotting of scatter graph

    Args:
    data_x: list or np.array, the list of x-values to display
    data_y: list or np.array, the list of y-values to display
    title: str, the title of the plot
    label_x: str, the x-axis label
    label_y: str, the y-axis label
    figsize: pair of ints, size of output plot
    dpi:int, dpi of output plot
    output_path: str or None, if None then show graph, otherwise save it
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(data_x, data_y)
    plt.title(title, fontsize=20)
    plt.xlabel(label_x, fontsize=16)
    plt.ylabel(label_y, fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Display or save
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_2D_line(
        data_x,
        data_y,
        title,
        label_x,
        label_y,
        figsize,
        dpi,
        output_path=None
        ):
    """
    Abstracted plotting of line graph

    Args:
    data_x: list or np.array, the list of x-values to display
    data_y: list or np.array, the list of y-values to display
    title: str, the title of the plot
    label_x: str, the x-axis label
    label_y: str, the y-axis label
    figsize: pair of ints, size of output plot
    dpi:int, dpi of output plot
    output_path: str or None, if None then show graph, otherwise save it
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(data_x, data_y)
    plt.title(title, fontsize=20)
    plt.xlabel(label_x, fontsize=16)
    plt.ylabel(label_y, fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Display or save
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
