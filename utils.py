import numpy as np

DX = 1E-6

def clean_vector(vector, length=3):
    """
    Ensure a vector is of the right length and format before setting it
    """
    vector = np.array(vector)
    if len(vector) != length:
        raise ValueError(f'Vector should have length {length}')
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
    return (f(t, delta_plus) - f(t, delta_minus))/delta
