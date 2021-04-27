"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Classes:

Simulation: Holds information for a simulation and allows multiprocessing
evaluation of particle integrators

Functions:
_integ: Used to call integration method of a particle in multiprocessing
load_pickle: Wrapper to easily load a pickle binary
"""

from USP import trap as USP_trap
from USP import particle
from USP import utils
from USP import consts
from USP import figs
import numpy as np
from itertools import repeat
import multiprocessing as mp
import pickle
import time
import csv
import warnings
from textwrap import dedent

from sys import getsizeof


def _integ(particle, potential, events):
    """
    Call init_integ of a passed mol. For parallelisation and handling
    integration errors
    """
    try:
        particle.integ(potential, events=events)
    except Exception as e:
        particle._terminate(particle._t_0)
        print(f'Integration did note complete for a particle starting at {particle.Q(particle._t_0)}')
        print(e)
    return particle

def load_pickle(filename):
    """
    Load a pickled simulation object and return it

    Args:
    filename
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


class Simulation:
    """
    Set up a simulation for particles in a specified trap over a given time
    period.
    """

    def __init__(self,
                 trap,
                 t_0,
                 t_end,
                 dt,
                 sample_points,
                 events = None,
                 process_no=None
                 ):
        """
        Simulation constructor:

        Args:
        trap: USP.trap object, provides potential for the simulation
        (trap.potential method)
        t_0: float, start time
        t_end: float, end time
        dt: float, timestep,
        sample_points: int, no. of points to sample in the simulation
        events: (list of) callable: events to be passed to integrator (default
        None)
        process_no: int or None, number of parallel processes or if None
        (default) uses maximum possible
        """
        self.set_trap(trap)
        self._t_0 = t_0
        self._t_end = t_end
        self._dt = dt
        self._sample_points = sample_points
        self._events = events
        self._process_no = process_no
        self._sample_times = np.linspace(t_0, t_end, sample_points)
        self._particles = None
        self.run_time = None

    @property
    def t_0(self):
        return self._t_0

    @property
    def t_end(self):
        return self._t_end

    @property
    def dt(self):
        return self._dt

    @property
    def points(self):
        return self._sample_points

    @property
    def times(self):
        """
        Alias self._sample_times, the times at which results will be sampled
        """
        return self._sample_times

    @property
    def particles(self):
        self._check_particles()
        return self._particles

    @property
    def result(self):
        """
        TODO: May be helpful to provide some sort of reduced form of output for
        exporting
        """
        raise NotImplemented

    def live_particles(self, t):
        """
        Return the list of all particles that have not been terminated at the
        given time

        Args:
        t: float, time of evaluating particle termination
        """
        return [p for p in self.particles if p.terminated > t]

    def terminated_particles(self, t, index=True):
        """
        Return the list of all particles that have been terminated at the given
        time

        Args:
        t: float, time of evaluating particle termination
        """
        return [p for p in self.particles if p.terminated < t]

    def _check_particles(self, message=None):
        """
        Raise runtime error if user tries to access particles before
        initialisation
        """
        if message is None:
            message = 'No particles have yet been initialised'
        if self._particles is None:
            raise RuntimeError(message)

    def set_trap(self, trap):
        """
        Set trap, checking type in the process. Can also set trap to None.
        """
        if trap is None:
            self._trap = None
        elif isinstance(trap, USP_trap.AbstractTrap):
            self._trap = trap
        elif isinstance(trap, USP_trap.AbstractPotentialTrap):
            self._trap = trap
        else:
            error = '''trap is not of valid type, expected None, AbstractTrap
            or AbstractPotentialTrap'''
            raise ValueError(error)

    def get_rs(self, t, live=True):
        """
        Return a list of all particles positions at the given time

        Args:
        t: float, time of evaluation
        live: bool, if true then return only the positions of live particles
        """
        if live:
            particles = self.live_particles(t)
        else:
            particles = self.particles
        rs = [p.r(t) for p in particles]
        return np.array(rs)

    def get_vs(self, t, live=True):
        """
        Return a list of all particles velocities at the given time

        Args:
        t: float, time of evaluation
        live: bool, if true then return only the positions of live particles
        """
        if live:
            particles = self.live_particles(t)
        else:
            particles = self.particles
        vs = [p.v(t) for p in particles]
        return np.array(vs)

    def get_ps(self, t):
        """
        Return a list of all particle momenta at the given time

        Args:
        t: float, time of evaluation
        """
        ps = [p.v(t) * p.m for p in self._particles]
        return np.array(ps)

    def get_KEs(self, t):
        """
        Return a list of the kinetic energy for each paticle at given time

        Args:
        t: float, time of evaluation
        """
        kinetics = [p.kinetic_energy(t) for p in self._particles]
        return kinetics

    def temperature(self, t):
        """
        Return the temperature of the cloud calculated from the KE at a given
        time

        Args:
        t: float, time of evaluation
        """
        kinetics = [p.kinetic_energy(t) for p in self.particles]
        mean_kinetic = np.mean(np.array(kinetics))
        return 2 * mean_kinetic / (3 * consts.k_B)

    def particle_number(self, t):
        """
        Return the number of live particles in the simulation at time

        Args:
        t: float, time of evaluation
        """
        return len(self.live_particles(t))

    def center(self, t):
        """
        Return the cloud centre as a numpy array at given time

        Args:
        t: float, time of evaluation
        """
        rs = self.get_rs(t)
        rs_T = rs.transpose()
        x_mean = np.mean(rs_T[0])
        y_mean = np.mean(rs_T[1])
        z_mean = np.mean(rs_T[2])
        return np.array([x_mean, y_mean, z_mean])

    def width(self, t):
        """
        Return the width of the cloud in the cardinal directions as a numpy
        array

        Args:
        t: float, time of evaluation
        """
        rs = self.get_rs(t)
        rs_T = rs.transpose()
        x_std = np.std(rs_T[0])
        y_std = np.std(rs_T[1])
        z_std = np.std(rs_T[2])
        return np.array([x_std, y_std, z_std])

    def velocity_width(self, t):
        """
        Return the width in velocity space of the cloud in the cardinal
        directions as a numpy array

        Args:
        t: float, time of evaluation
        """
        vs = np.array(self.get_vs(t))
        vs_T = vs.transpose()
        v_x_std = np.std(vs_T[0])
        v_y_std = np.std(vs_T[1])
        v_z_std = np.std(vs_T[2])
        return np.array([v_x_std, v_y_std, v_z_std])

    def momentum_width(self, t):
        """
        Return the width in momentum space of the cloud in the cardinal
        directions as a numpy array

        Args:
        t: float, time of evaluation
        """
        ps = self.get_ps(t)
        ps_T = ps.transpose()
        p_x_std = np.std(ps_T[0])
        p_y_std = np.std(ps_T[1])
        p_z_std = np.std(ps_T[2])
        return np.array([p_x_std, p_y_std, p_z_std])

    def pickle(self, path=None, prepend_path=''):
        """
        Save the simulation object as a pickle.

        Args:
        path: the path at which to store the pickle file, default None, in
        which case use time
        prepend_path: string to prepend to the path (e.g. for adding a
        directory but using the datetime as the filename)
        """
        if path is None:
            path = time.strftime("%Y-%m-%d--%H-%M-%S.pickle")
        else:
            path = str(path)
        path = prepend_path + path
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def run(self):
        """
        Run the simulation using multiprocessing.
        """
        # Check that a trap has been specified
        if self._trap is None:
            raise RuntimeError(
                    'No trap has been specified for this simulation'
                    )

        # Create args for _integ running in parallel
        args = zip(
                self._particles,
                repeat(self._trap.potential),
                repeat(self._events)
                )

        # Check processes available
        available = mp.cpu_count()
        if self._process_no is None:
            process_no = available
        elif self._process_no > available:
            message = f'''
                Requested process no. ({self._process_no}) exceeds available
                cores ({available}). Running with available cores.
                '''
            warnings.warn(message, RuntimeWarning)
            process_no = available
        else:
            process_no = self._process_no

        # Announce
        print(dedent(f'''
                Running simulation
                Particles: {self._N_particles}
                Processes: {process_no}
                Time: {self._t_end - self._t_0}
                dt: {self._dt}
                '''
                     )
              )

        # Multiprocessing
        start_time = time.time()
        multiprocessing_flag = True # TODO Enable turning this to False
        if multiprocessing_flag:
            with mp.Pool(process_no) as p:
                self._particles = p.starmap(_integ, args)
        else:
            self._particles = [
                    p.integ(self._trap.potential, events=self._events)
                    for p in self._particles
                    ]
        self.run_time = time.time() - start_time

        print(f'Complete, runtime: {self.run_time:.3f}')

    # TODO Initialisation parameters should be passed to init, and this
    # function called automatically unless flagged not to
    def init_particles(
            self,
            particle_no,
            mass,
            r_generator,
            v_generator
            ):
        """
        Create particles for simulation.

        Args:
        particle_no: int
        mass: float
        r_generator: generate random positions
        v_generator: generate random velocities

        Output:
        None, populates self._particles
        """


        self._mass = mass  # Save for use in temperature calculations see TODO

        # Generate particles
        self._particles = [
            particle.Particle(
                r_generator(),
                v_generator(),
                mass,
                self._t_0,
                self._t_end,
                self._dt,
                self._sample_points
              )
            for i in range(int(particle_no))
            ]

        # Save no. of particles for reference
        self._N_particles = particle_no

    def clean_particles(self):
        """
        Remove any particles without integrators
        """
        self._particles = [p for p in self.particles if p._result is not None]

    def get_total_energy(self, t):
        """
        Get the total energy (KE + potential) at a specified time

        Args:
        t: float, time at which to retrieve Q
        """
        energies = [p.energy(t, self._trap.potential) for p in self._particles]
        return np.sum(np.array(energies))
