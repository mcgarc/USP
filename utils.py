import numpy as np


def clean_vector(vector, length=3):
    """
    Ensure a vector is of the right length and format before setting it
    """
    vector = np.array(vector)
    if len(vector) != 3:
        raise ValueError(f'Vector should have length {length}')
    return vector
