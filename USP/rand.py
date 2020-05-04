import numpy as np

class Generator:

    def generate(self):
        raise NotImplementedError

    def __call__(self):
        return self.generate()

class UniformGenerator(Generator):

    def __init__(low, high, length):
        self._low = low
        self._high = high
        self._length = length
        # TODO check center and spread correspond in dimension
        self._scalar = np.isscalar(low)


    def generate(self):
        if self._scalar:
            return np.random.uniform(low, high, length)
        else:
            result = [np.random.uniform(self._low[i], self._high[i])
                    for i in range(self._length)]
            return np.array(result)


class NormalGenerator(Generator):

    def __init__(self, center, spread, length=1):
        self._center = center
        self._spread = spread
        self._scalar = np.isscalar(center)
        if self._scalar:
            self._length = length
        else:
            # TODO check center and spread correspond in dimension
            self._length = len(self._center)

    def generate(self):
        if self._scalar:
            return np.random.normal(self._center, self._spread, self._length)
        else:
            result = [np.random.normal(self._center[i], self._spread[i])
                    for
                    i
                    in range(self._length)
                    ]
            return np.array(result)
