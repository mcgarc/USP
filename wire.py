import numpy as np


class WireSegment:
    """
    Simulates a segment of straight wire
    """

    def __init__(self, start, end, current):
        self.set_start(start)
        self.set_end(end)
        self.set_current(current)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def current(self):
        return self._current

    def _clean_vector(self, vector, length=3):
        """
        Ensure a vector is of the right length and format before setting it
        """
        vector = np.array(vector)
        if len(vector) != 3:
            raise ValueError(f'Vector should have length {length}')
        return vector

    def set_start(self, start):
        self._start = self._clean_vector(start)

    def set_end(self, end):
        self._end = self._clean_vector(end)

    def set_current(self, current):
        self._current = current

