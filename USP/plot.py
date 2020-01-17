from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

def get_row_r(row):
    return [float(x) for x in row[:3]]

def plot_positions(
        filenames,
        colours,
        x_lim = None,
        y_lim = None,
        z_lim = None
        ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # TODO check filenames and colors same length
    no_plots = len(filenames)
    
    for i in range(no_plots):
        filename = filenames[i]
        colour = colours[i]
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            positions = [get_row_r(row) for row in reader]
        positions = np.array(positions).transpose()
        xs = positions[0,:]
        ys = positions[1,:]
        zs = positions[2,:]
        ax.scatter(xs, ys, zs, colour)

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if z_lim is not None:
        ax.set_zlim(*z_lim)
    plt.show()

    



if __name__ == '__main__':
    plot_positions(
            ['output/0.0.csv', 'output/1000.0.csv'],
            ['b', 'r']
            )
