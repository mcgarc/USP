import numpy as np

from . import utils

class StaticField:
    """
    Return a field of constant value
    """

    def __init__(self, field_array):
        self.field_array = np.array(field_array)

    def field(self, t, r):
        return self.field_array
        
class QuadrupoleField:
    """
    Approximation of a quadrupole field close to the centre
    """

    def __init__(self, b_1, r_0=[0,0,0]):
        """
        Take the field gradient b_1 as an argument. Optional trap centre arg r_0
        """
        self.b_1 = b_1
        self.r_0 = utils.clean_vector(r_0)

    def field(self, t, r):
        r = np.array(r) - self.r_0
        scale = np.array([-0.5, -0.5, 1])
        return self.b_1 * scale * r
