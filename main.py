import numpy as np


class Particle:
    """
    Contain information about a particle
    """

    def __init__(self, r, v, m):
        self.set_r(r)
        self.set_v(v)
        self.set_m(m)

    @property
    def r(self):
        return self._r

    @property
    def v(self):
        return self._v

    @property
    def m(self):
        return self._m

    def _clean_vector(self, vector, length=3):
        """
        Ensure a vector is of the right length and format before setting it
        """
        vector = np.array(vector)
        if len(vector) != 3:
            raise ValueError(f'Vector should have length {length}')
        return vector

    def set_r(self, r):
        self._r = self._clean_vector(r)

    def set_v(self, v):
        self._v = self._clean_vector(v)

    def set_m(self, m):
        self._m = m


