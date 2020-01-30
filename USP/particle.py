import numpy as np
from itertools import count
import desolver as de
import desolver.backend as D

### FIXME This is only the second hackiest bit of code I have ever written...
# Fixes bug with desolver trying to cast lists/ arrays/ ndarrays as floats
def float_conv(x):
    if type(x) is list:
        x = np.array(x)
    if type(x) in [D.array, np.ndarray]:
        return x.astype(float)
    else:
        return float(x)

D.to_float = float_conv
###

from . import utils


class Particle:
    """
    Describes a particle with spatial position, velocity and mass. Particle also
    has the attribute _integ, which is the integrator to describe its propagation
    in time, and the attribute _terminated, which flags if the particle has
    been destroyed or otherwise lost
    """
    _index_count = count(0)

    def __init__(self, r, v, m):
        self.set_r(r)
        self._r_0 = self.r
        self.set_v(v)
        self._v_0 = self.v
        self.set_m(m)
        self._integ = None
        self._terminated = False
        self._index = next(self._index_count)

    @property
    def m(self):
        return self._m

    @property
    def terminated(self):
        return self._terminated

    @property
    def index(self):
        return self._index

    def Q(self, t):
        if self._integ is None:
            if t != 0:
                raise ValueError('No integrator, so cannot provide Q(t>0)')
            else:
                return np.concatenate((self._r, self._v)) # Q(t=0)
        else:
            return self._integ[t][1] # 1 gets y part

    def r(self, t):
        return self.Q(t)[:3]

    def v(self, t):
        return self.Q(t)[3:]

    def set_r(self, r):
        self._r = utils.clean_vector(r)

    def set_v(self, v):
        self._v = utils.clean_vector(v)

    def set_m(self, m):
        self._m = m

    def terminate(self):
        self._terminated = True

    def kinetic_energy(self, t):
        return 0.5 * self.m * np.dot(self.v(t), self.v(t))

    def potential_energy(self, t, potential):
        """
        Return the potential energy of the particle at time t

        Args:
        t: float, the time at which the potential should be evaluated
        potential: of form `USP.trap.potential`
        """
        return potential(t, self.r(t))

    def energy(self, t, potential=None):
        """
        Return the total energy of the particle. If no potential is provided
        then just take the KE

        Args:
        t: float, the time at which the energy should be evaluated
        potential: of form `USP.trap.potential` or None
        """
        energy = self.kinetic_energy(t)
        if potential is not None:
            energy += self.potential_energy(t, potential)
        return energy

    def integ(self, potential, t_0, t_end, dt):
        """
        Create and solve initial value problem for the particle. Takes the
        potential (which we expect to be a function of t and r) and a boundary
        time. Can take the initial time (assumed zero) and a maximum allowed
        step (assumed inf)
        """
        # Construct dQ_dt, which is to be stepped by the integrator
        integ = de.OdeSystem(
                self._dQ_dt,
                y0 = D.array(self.Q(0)),
                dense_output = False,
                t = (t_0, t_end),
                dt = dt,
                constants = {'potential': potential}
                )
        integ.set_method('SymplecticEulerSolver')
        integ.integrate()
        self._integ = integ

    def _dQ_dt(self, t, Q, potential):
        """
        Return the time derivative of Q, the 6-vector position in momentum space,
        for a particle in given potential at time t. (Hamilton's equations)
        """
        r = Q[:3]
        v = Q[3:]
        dvx_dt = -utils.grad(potential, t, r, 'x') / self.m
        dvy_dt = -utils.grad(potential, t, r, 'y') / self.m
        dvz_dt = -utils.grad(potential, t, r, 'z') / self.m
        dQ_dt = np.array([
                v[0],
                v[1],
                v[2],
                dvx_dt,
                dvy_dt,
                dvz_dt
                ])
        return dQ_dt

    def check_termination(self):
        """
        TODO: Check if the particle must be terminated, for example due to
        violating some boundary condition (like touching the surface of a chip!)
        """
        pass
