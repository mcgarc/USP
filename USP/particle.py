"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Classes:

Particle: Holds information on a particle and its integrator
Trajectory: Holds information on a trajectory
"""

import numpy as np
from itertools import count
import desolver as de
import desolver.backend as D
from . import utils

# FIXME This is only the second hackiest bit of code I have ever written...
# Fixes bug with desolver trying to cast lists/ arrays/ ndarrays as floats


def float_conv(x):
    if type(x) is list:
        x = np.array(x)
    if type(x) in [D.array, np.ndarray]:
        return x.astype(float)
    else:
        return float(x)


D.to_float = float_conv

class Particle:
    """
    Describes a particle with spatial position, velocity and mass. Particle
    also has the attribute _integ, which is the integrator to describe its
    propagation in time, and the attribute _terminated, which flags if the
    particle has been destroyed or otherwise lost
    """
    _index_count = count(0)

    def __init__(
            self,
            r,
            v,
            m,
            t_0,
            t_end,
            dt,
            points
            ):
        """
        Initialise a particle with starting position, velocity and mass

        Args:
        r: list-like, initial position
        v: list-like, initial velocity
        m: float, mass
        t0: float, starting time
        t_end: float, end time
        dt: float, step, or None
        points: int, the number of points at which to save Q from integrator
        """
        self.set_r(r)
        self._r_0 = self.r
        self.set_v(v)
        self._v_0 = self.v
        self.set_m(m)
        self._t_0 = t_0
        self._t_end = t_end
        self._dt = dt
        self._points = points
        self._result = None
        # Start with no termination, i.e. time terminated is infinity
        self._terminated = np.inf
        self._index = next(self._index_count)
        self._result_times = np.linspace(t_0, t_end, points)

    @property
    def m(self):
        return self._m

    @property
    def terminated(self):
        return self._terminated

    @property
    def index(self):
        return self._index

    @property
    def result_times(self):
        """
        np.linspace containing the times at which we sample particle Q values
        """
        return np.linspace(self._t_0, self._t_end, self._points)

    def Q(self, t):
        """
        Return Q = [r, v] at the closest time t to that which was sampled by
        the results

        Args:
        t: float, time
        """
        if self._result is None:
            if t != 0:
                raise ValueError('No integrator, so cannot provide Q(t>0)')
            else:
                return np.concatenate((self._r, self._v))  # Q(t=0)
        else:
            t_index = self._get_time_index(t)
            return self._result[t_index]

    def _get_time_index(self, t):
        """
        Returns the index of the time closest to the input time. Used to
        address the result list

        Args:
        t: float, time
        """
        index = 0
        # Check that we are in bounds
        if t < self._t_0 or t > self._t_end:
            raise ValueError('Time specification out of bounds for particle')
        # Otherwise iterate
        for sample_time in self._result_times:
            if t <= sample_time:
                return index
            index += 1
        # If somehow we get here then set index to -1
        return -1

#    def _index_to_time(self, time_index):
#        """
#        Convert a time index (int) into an absolute time (float) based on the
#        number of samples
#
#        Args:
#        time_index: int, the time index to be converted
#        """
#        Dt = (self._t_end - self._t_0) / self._sample_points
#        return time_index * Dt

    def r(self, t):
        return self.Q(t)[:3]

    def v(self, t):
        return self.Q(t)[3:]

    def speed(self, t):
        return np.linalg.norm(self.v(t))

    def Q_projection(self, t, dir_index):
        """
        Return the projection of Q onto the ith 2D plane, i.e. return the
        vector [r_i, v_i] at time t

        Arg:
        t: float, time
        dir_index: int, 0, 1, 2 corresponds to x, y and z respectvely
        """
        i = utils.clean_direction_index(dir_index)
        return np.array([self.r(t)[i], self.v(t)[i]])

    def set_r(self, r):
        self._r = utils.clean_vector(r)

    def set_v(self, v):
        self._v = utils.clean_vector(v)

    def set_m(self, m):
        self._m = m

    def _terminate(self, t):
        """
        Set the termination attribute to time of termination

        Args:
        t: float, the time of the termination
        """
        self._terminated = float(t)

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

    def integ(self, potential, events=None):
        """
        Create and solve initial value problem for the particle. Takes the
        potential (which we expect to be a function of t and r) and a boundary
        time.

        Args:
        potential: method, with args (t, r). Passed to _dQ_dt. e.g. a
        `trap.potential`
        events: (list of) callable: event method(s) to be passed to Desolver
        integrator
        """
        # Initialise Desolver integrator
        integ = de.OdeSystem(
                self._dQ_dt,
                y0=D.array(self.Q(0)),
                dense_output=True,
                t=(self._t_0, self._t_end),
                dt=self._dt,
                constants={'potential': potential}
                )
        integ.set_method('RK45')
        integ.integrate(events=events)
        # Look for early completion by comparing last element of evaluation
        # times to desired end time. Use isclose because integ can be off by
        # fp error
        integ_end_time = integ.t[-1]
        if not np.isclose(integ_end_time, self._t_end):
            self._terminate(integ_end_time)
        # Post-processing of resulting trajectory
        self._result = [integ[float(t)][1] for t in self.result_times]

    def _dQ_dt(self, t, Q, potential):
        """
        Return the time derivative of Q, the 6-vector position in momentum
        space, for a particle in given potential at time t. (Hamilton's
        equations)

        Args:
        t: float, time of evaluation
        Q: list-like: numerical evaluation of Q=[r,v] at time t
        potential: method, a method `trap.potential` which takes arguments t, r
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
        violating some boundary condition (like touching the surface of a
        chip!)
        """
        pass
