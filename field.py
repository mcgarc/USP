import numpy as np

class StaticField:
    """
    Return a field of constant value
    """

    def __init__(self, field_array):
        self.field_array = np.array(field_array)

    def field(self, t, r):
        return self.field_array
        
