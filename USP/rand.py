import numpy as np
from USP import consts

class Generator:

    def generate(self):
        raise NotImplementedError

    def __call__(self):
        return self.generate()

class UniformGenerator(Generator):

    def __init__(self, low, high, length=1):
        self._low = low
        self._high = high
        self._scalar = np.isscalar(low)
        if self._scalar:
            self._length = length
        else:
            # TODO check low and high correspond in dimension
            self._length = len(self._low)


    def generate(self):
        if self._scalar:
            return np.random.uniform(self._low, self._high, self._length)
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

class TemperatureGenerator(UniformGenerator):
    """
    Takes input in temperature and creates a unifrom distribution in
    temperature space, but output is in m/s for direct input to
    Simulation.init_particles
    """

    def __init__(self, mass, low, high=None, length=1):
        """
        Initialise temperature generator. If only lower bound is supplied then
        take this to be the (negative of) upper (lower) bound.
        args:
        mass: mass of particle in kg
        low: the lower bound, or if high is None, half the spread about zero
        (in Kelvin)
        high: the upper bound (in Kelvin)
        length: the size of the array to generate
        """
        self._mass = mass # Required to convert to temperature
        if high is None:
            high = abs(low)
            low = -abs(low)
        super().__init__(low, high, length)

    def generate(self):
        """
        Generate in temperature space, but convert output into m/s for
        convenience. This means that the distribution is still uniform over
        temperature
        """
        temp = super().generate()
        vel = np.sqrt(2 * consts.k_B * np.abs(temp) / self._mass)
        # Keep signs
        vel = vel * np.sign(temp)
        return vel
