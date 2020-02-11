import numpy as np
import csv

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
    if direction == 'x' or direction == 0:
        i = 0
    elif direction == 'y' or direction == 1:
        i = 1
    elif direction == 'z' or direction == 2:
        i = 2
    else:
        raise ValueError('Unexpected gradient direction')
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
