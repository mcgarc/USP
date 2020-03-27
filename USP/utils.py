import numpy as np
import csv
from matplotlib import pyplot as plt

DX = 1E-6

def clean_vector(vector, length=3, cast_type=None):
    """
    Ensure a vector is of the right length and format before setting it
    """
    vector = np.array(vector)
    if len(vector) != length:
        raise ValueError(f'Vector should have length {length}')
    if cast_type is not None:
        vector = vector.astype(cast_type)
    return vector

def grad(f, t, r, direction, delta=DX):
    """
    Returns the gradient of a function f = f(t, r) in the specified direction
    """
    r = np.array(r)
    i = clean_direction_index(direction)
    direction = [0, 0, 0]
    direction[i] = 1
    delta_plus = r + delta * np.array(direction)
    delta_minus = r - delta * np.array(direction)
    return (f(t, delta_plus) - f(t, delta_minus))/(2*delta)

def create_output_csv(times, mol_Qs):
    """
    Procuce a CSV file for each time at which Q(t) was evaluated. Each CSV
    contains Q(t) for each molecule.

    Args
    times: list of times at which molecule Q(t) were evaluated
    mol_Qs: list of Q(t) values for each time and for each molecule
    """
    # Transpose to desired format
    mol_Qs = np.array(mol_Qs)
    time_Qs = mol_Qs.transpose(1,0,2)
    # Check times provided match Qs
    if len(times) != time_Qs.shape[0]:
        raise ValueError()
    for t_index in range(len(times)):
        time = times[t_index]
        Qs = time_Qs[t_index]

        with open(f'output/{time}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ' ')
            for Q in Qs:
                # TODO Index
                writer.writerow(Q)

def clean_direction_index(dir_index, str_rep=False):
    """
    Take a direction index and ensure it is valid. e.g. input 0, 1, 2
    corresponds to x, y and z directions. No other input allowed.

    Args:
    dir_index: int-castable or str, the direction to be cleaned
    str_rep: bool, whether to return the string representation of direction too
    """
    # Recast floats as ints
    if type(dir_index) in [float, np.float64]:
        dir_index = int(dir_index)
    # Get correct index
    if dir_index == 'x' or dir_index == 0:
        i = 0
        s = 'x'
    elif dir_index == 'y' or dir_index == 1:
        i = 1
        s = 'y'
    elif dir_index == 'z' or dir_index == 2:
        i = 2
        s = 'z'
    else:
        raise ValueError('Unexpected direction index')
    # Return index (and string representation)
    if str_rep:
        return i, s
    return i

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
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.scatter(data_x, data_y)
    ax.autoscale()
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
