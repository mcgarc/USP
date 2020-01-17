import numpy as np
import scipy.integrate as integ
from itertools import count

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
                raise ValueError('No integrator, so cannot provide r(t>0)')
            else:
                return np.concatenate((self._r, self._v)) # Q(t=0)
        else:
            return self._integ.sol(t)

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

    def integ(
            self,
            potential,
            t_end,
            t_0=0,
            max_step=np.inf # This is the regular default value
            ):
        """
        Create and solve initial value problem for the particle. Takes the
        potential (which we expect to be a function of t and r) and a boundary
        time. Can take the initial time (assumed zero) and a maximum allowed
        step (assumed inf)
        """
        # Construct dQ_dt, which is to be stepped by the integrator
        dQ_dt = lambda t, Q: self._dQ_dt(t, Q, potential)
        self._integ = integ.solve_ivp(
                dQ_dt,
                (t_0, t_end),
                self.Q(0),
                method='RK45',
                max_step=max_step,
                dense_output=True,
                vectorized=True
                )

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
