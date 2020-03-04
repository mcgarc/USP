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
"""

from USP import trap as USP_trap
from USP import particle
from USP import utils
from USP import consts
import numpy as np
from itertools import repeat
import multiprocessing as mp
import pickle
import time
import csv
from textwrap import dedent

from sys import getsizeof

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def _integ(particle, potential, events):
    """
    Call init_integ of a passed mol. For parallelisation
    """
    particle.integ(potential, events=events)
    return particle


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
        self._eval_times = np.arange(t_0, t_end, dt)
        self._particles = None
        self.run_time = None

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

    def live_particles(self, t_index):
        """
        Return the list of all particles that have not been terminated at the
        given time

        Args:
        t_index: int, time index of evaluating particle termination
        """
        t = self._index_to_time(t_index)
        return [p for p in self.particles if p.terminated > t]

    def terminated_particles(self, t, index=True):
        """
        Return the list of all particles that have been terminated at the given
        time

        Args:
        t_index: int, time index of evaluating particle termination
        """
        t = self._index_to_time(t_index)
        return [p for p in self.particles if p.terminated < t]

    def _index_to_time(self, time_index):
        """
        Convert a time index (int) into an absolute time (float) based on the
        number of samples

        Args:
        time_index: int, the time index to be converted
        """
        Dt = (self._t_end - self._t_0) / self._sample_points
        return time_index * Dt

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
        t: int, index time of evaluation
        live: bool, if true then return only the positions of live particles
        """
        if live:
            particles = self.live_particles(t)
        else:
            particles = self.particles
        rs = [p.r(t) for p in particles]
        return rs

    def get_vs(self, t, live=True):
        """
        Return a list of all particles velocities at the given time

        Args:
        t: int, index time of evaluation
        live: bool, if true then return only the positions of live particles
        """
        if live:
            particles = self.live_particles(t)
        else:
            particles = self.particles
        vs = [p.v(t) for p in particles]
        return vs

    def get_ps(self, t):
        """
        Return a list of all particle momenta at the given time

        Args:
        t: int, index time of evaluation
        """
        ps = [p.v(t) * p.m for p in self._particles]
        return ps

    def get_KEs(self, t):
        """
        Return a list of the kinetic energy for each paticle at given time

        Args:
        t: int, index time of evaluation
        """
        kinetics = [p.kinetic_energy(t) for p in self._particles]
        return kinetics

    def temperature(self, t):
        """
        Return the temperature of the cloud calculated from the KE at a given
        time

        Args:
        t: int, index time of evaluation
        """
        kinetics = [p.kinetic_energy(t) for p in self.particles]
        mean_kinetic = np.mean(np.array(kinetics))
        return 2 * mean_kinetic / (3 * consts.k_B)

    def particle_number(self, t):
        """
        Return the number of live particles in the simulation at time

        Args:
        t: int, index time of evaluation
        """
        return len(self.live_particles(t))

    def center(self, t):
        """
        Return the cloud centre as a numpy array at given time

        Args:
        t: int, index time of evaluation
        """
        rs = np.array(self.get_rs(t))
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
        t: int, index time of evaluation
        """
        rs = np.array(self.get_rs(t))
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
        t: int, index time of evaluation
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
        t: int, index time of evaluation
        """
        ps = np.array(self.get_ps(t))
        ps_T = ps.transpose()
        p_x_std = np.std(ps_T[0])
        p_y_std = np.std(ps_T[1])
        p_z_std = np.std(ps_T[2])
        return np.array([p_x_std, p_y_std, p_z_std])

    # TODO Improve output/ fix
    def save_sim_info(self, filename):
        """
        Save information on the string

        Args:
        filename: str, file to save csv
        """
        output = f'''
        Start: {self._t_0}
        End: {self._t_end}
        No. of particles: {self._N_particles}
        '''
        with open(filename, 'w') as f:
            f.write(self._particles, f)

    def save_Q_to_csv(self, t, filename):
        """
        Save Q(t) for each particle as CSV

        Args:
        t: int, time index at which to retrieve Q
        filename: str, file to save csv
        """

        with open(filename, 'w') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            for p in self._particles:
                row = [p.index] + list(p.Q(t))
                wr.writerow(row)

    def save_to_pickle(self, filename):
        """
        Save particles as a pickle
        """
        with open(filename, 'wb') as f:
            pickle.dump(self._particles, f)

    def load_from_pickle(self, filename):
        """
        Load particles from pickle
        """
        with open(filename, 'rb') as f:
            self._particles = pickle.load(f)

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
            raise RuntimeWarning(
                f'''
                Requested process no. ({self._process_no}) exceeds available
                cores ({available}). Running with available cores.
                '''
                )
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
        with mp.Pool(process_no) as p:
            self._particles = p.starmap(_integ, args)
        self.run_time = time.time() - start_time

        print(f'Complete, runtime: {self.run_time:.3f}')

    # TODO Initialisation parameters should be passed to init, and this
    # function called automatically unless flagged not to
    def init_particles(
            self,
            particle_no,
            mass,
            r_sigma,
            v_sigma,
            r_centre=[0, 0, 0],
            v_centre=[0, 0, 0],
            seed=None
            ):
        """
        Create particles for simulation.

        Args:
        particle_no: int
        mass: float
        r_sigma: std for particle position distribution
        v_sigma: std for particle velocity distribution
        r_centre: where to centre positions (default origin)
        v_centre: where to centre velocities (default origin)
        seed: if not None, sets the numpy seed

        Output:
        None, populates self._particles
        """

        self._mass = mass  # Save for use in temperature calculations see TODO

        if seed is not None:
            np.seed(seed)

        # Clean sigma input into np arrays of length 3
        if type(r_sigma) in [float, int, np.float64]:
            r_sigma = [r_sigma, r_sigma, r_sigma]
        if type(v_sigma) in [float, int, np.float64]:
            v_sigma = [v_sigma, v_sigma, v_sigma]
        r_sigma = utils.clean_vector(r_sigma)
        v_sigma = utils.clean_vector(v_sigma)

        # Clean centre inputs into np arrays of length 3
        r_centre = utils.clean_vector(r_centre)
        v_centre = utils.clean_vector(v_centre)

        # Generate particles
        self._particles = [
            particle.Particle(
              [
                  np.random.normal(r_centre[0], r_sigma[0]),
                  np.random.normal(r_centre[1], r_sigma[1]),
                  np.random.normal(r_centre[2], r_sigma[2])
              ],
              [
                  np.random.normal(v_centre[0], v_sigma[0]),
                  np.random.normal(v_centre[1], v_sigma[1]),
                  np.random.normal(v_centre[2], v_sigma[2])
              ],
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

    def get_total_energy(self, t):
        """
        Get the total energy (KE + potential) at a specified time

        Args:
        t: int, time index at which to retrieve Q
        """
        energies = [p.energy(t, self._trap.potential) for p in self._particles]
        return np.sum(np.array(energies))

    def plot_start_end_positions(self):
        """
        3D plot of inital and final positions in the simulation
        """
        start_rs = np.array(self.get_rs(0)).transpose()
        end_rs = np.array(self.get_rs(-1)).transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(start_rs[0], start_rs[1], start_rs[2], 'b')
        ax.scatter(end_rs[0], end_rs[1], end_rs[2], 'r')
        plt.show()

    def _plot_2D_scatter(
            self,
            data_x,
            data_y,
            title,
            label_x,
            label_y,
            figsize,
            dpi,
            output_path=None
            ):
        """
        Abstracted plotting of scatter graph

        Args:
        data_x: list or np.array, the list of x-values to display
        data_y: list or np.array, the list of y-values to display
        title: str, the title of the plot
        label_x: str, the x-axis label
        label_y: str, the y-axis label
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        output_path: str or None, if None then show graph, otherwise save it
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(data_x, data_y)
        plt.title(title, fontsize=24)
        plt.xlabel(label_x, fontsize=20)
        plt.ylabel(label_y, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Display or save
        if output_path is not None:
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def _plot_histogram(
            self,
            data,
            bins,
            title,
            label_x,
            label_y,
            figsize,
            dpi,
            output_path
            ):
        """
        Abstracted plotting of histogram

        Args:
        data: list or np.array, the data to plot in the histogram
        bins: int, the number of bins in the histogram
        title: str, the title of the plot
        label_x: str, the x-axis label
        label_y: str, the y-axis label
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        output_path: str or None, if None then show graph, otherwise save it
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.hist(data, bins=bins)
        plt.title(title, fontsize=24)
        plt.xlabel(label_x, fontsize=20)
        plt.ylabel(label_y, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Display or save
        if output_path is not None:
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_particle_number(
            self,
            output_path=None,
            figsize=(6, 4),
            dpi=300
            ):
        """
        Plot the particle number at the sample points

        Args:
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        nos = [self.particle_number(t) for t in range(self._sample_points)]
        # Plot
        self._plot_2D_scatter(
                times,
                nos,
                f'Particle number',
                'time (s)',
                'Particle number',
                figsize,
                dpi,
                output_path
                )

    def plot_temperatures(
            self,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            ):
        """
        Plot the temperature at the sample points

        Args:
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        temps = [1E6 * self.temperature(t) for t in range(self._sample_points)]
        # Plot
        self._plot_2D_scatter(
                times,
                temps,
                f'Cloud temperature',
                'time (s)',
                'temperature (uK)',
                figsize,
                dpi,
                output_path
                )

    def plot_width(
            self,
            direction,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            ):
        """
        Plot the cloud width for given direction for each sample point

        Args:
        direction: int or str, indication of direction (x, y or z) to take
        width
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        direction, dir_label = utils.clean_direction_index(direction, True)
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        widths = [1E3 * self.width(t)[direction]
                  for t
                  in range(self._sample_points)
                  ]
        # Plot
        self._plot_2D_scatter(
                times,
                widths,
                f'Cloud width in {dir_label}',
                'time (s)',
                'width (mm)',
                figsize,
                dpi,
                output_path
                )

    def plot_velocity_width(
            self,
            direction,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            ):
        """
        Plot the cloud momentum width for given direction for each sample point

        Args:
        direction: direction index along which to plot centre
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        direction, dir_label = utils.clean_direction_index(direction, True)
        widths = [self.velocity_width(t)[direction]
                  for t
                  in range(self._sample_points)
                  ]
        # Plot
        self._plot_2D_scatter(
                times,
                widths,
                f'Velocity width in {dir_label}',
                'time (s)',
                'width (m/s)',
                figsize,
                dpi,
                output_path
                )

    def plot_center(
            self,
            direction,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            dist_unit='m'
            ):
        """
        Plot the mean position of the cloud at each sample point at some given
        direction

        Args:
        direction: direction index along which to plot centre
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        direction, dir_label = utils.clean_direction_index(direction, True)
        centers = [self.center(t)[direction]
                   for t
                   in range(self._sample_points)
                   ]
        # Plot
        self._plot_2D_scatter(
                times,
                centers,
                f'Cloud centre position along {dir_label} direction',
                'time (s)',
                f'{dir_label} ({dist_unit})',
                figsize,
                dpi,
                output_path
                )

    def plot_cloud_volume(
            self,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            dist_unit='m'
            ):
        """
        Plot the volume of the cloud as a function of time

        Args:
        N_points: int, no. of points to plot
        direction: direction index along which to plot centre
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        vols = [1E9 * np.prod(self.width(t))
                for t
                in range(self._sample_points)
                ]
        # Plot
        self._plot_2D_scatter(
                times,
                vols,
                f'Cloud volume',
                'time (s)',
                f'Volume (mm^3)',
                figsize,
                dpi,
                output_path
                )

    def plot_cloud_phase_space_volume(
            self,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            dist_unit='m'
            ):
        """
        Plot the phase space volume of the cloud

        Args:
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        dist_unit: str, representation of distance unit for labelling
        """
        # Get data
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        vols = [1E9 * np.prod(self.width(t)) * np.prod(self.momentum_width(t))
                / consts.u**3
                for t
                in range(self._sample_points)
                ]
        # Plot
        self._plot_2D_scatter(
                times,
                vols,
                f'Cloud PS volume',
                'time (s)',
                f'PS volume ((atomic mass * mm)^3)',
                figsize,
                dpi,
                output_path
                )

    def plot_position_histogram(
            self,
            time,
            direction,
            bins=20,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            ):
        """
        Plot a histogram of particle positions at given time, projected into
        one cardinal direction 

        Args:
        time: int, the time index at which to plot
        direction: direction index along which to plot centre
        bins: int, the number of bins in the histogram
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        direction, dir_label = utils.clean_direction_index(direction, True)
        data = 1000 * np.array(self.get_rs(time)).transpose()[direction]
        # Plot
        self._plot_histogram(
                data,
                bins,
                f'Position distribution in {dir_label}',
                f'{dir_label} (mm)',
                'frequency',
                figsize,
                dpi,
                output_path
                )

    def plot_momentum_histogram(
            self,
            time,
            direction,
            bins=20,
            output_path=None,
            figsize=(6, 4),
            dpi=300,
            ):
        """
        Plot a histogram of particle momenta at given time, projected into
        one cardinal direction 

        Args:
        time: int, the time index at which to plot
        direction: direction index along which to plot centre
        bins: int, the number of bins in the histogram
        output_path: str or None, if None then show graph, otherwise save it
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        """
        # Get data
        direction, dir_label = utils.clean_direction_index(direction, True)
        data = np.array(self.get_ps(time)).transpose()[direction]
        # Plot
        self._plot_histogram(
                data,
                bins,
                f'Momentum distribution in {dir_label}',
                f'{dir_label} momentum (kgm/s)',
                'frequency',
                figsize,
                dpi,
                output_path
                )

    def plot_phase_diagram(
            self,
            particle_index,
            dir_index,
            output_path=None,
            time_gradient=False,
            colorbar=False,
            figsize=(6, 4),
            dpi=300,
            **plot_kwargs
            ):
        """
        Plot the phase space diagram of a particle in the trap over time.

        Args:
        particle_index: int, the paticle whose PSD is to be shown
        dire_index: int, corresponds to the x, y or z direction
        output_path: str or None, if None then show graph, otherwise save it
        time_gradient: bool, if true then shade the points for time
        colorbar: bool, if time_gradient is true then colorbar will create a
            colour legend for the times
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        plot_kwargs: kwargs to be passed to the scatter plot fn
        """
        # Get coordinates
        times = np.linspace(self._t_0, self._t_end, self._sample_points)
        particle = self._particles[particle_index]
        Q_projections = [particle.Q_projection(t, 0)
                         for t
                         in range(self._sample_points)
                         ]
        Q_projections = np.array(Q_projections).transpose()
        # Setup time gradient if required
        if time_gradient:
            plot_kwargs['c'] = times
        # Create plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.05, 0.05, 0.8, 0.8])
        cm = plt.get_cmap('winter')
        sc = ax.scatter(Q_projections[0],
                        Q_projections[1],
                        cmap=cm,
                        **plot_kwargs
                        )
        # Colorbar
        if time_gradient and colorbar:
            cbar = plt.colorbar(sc)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('time', fontsize=20)
        # Label axes
        dir_label = {0: 'x', 1: ' y', 2: 'z'}[dir_index]
        plt.title(f'Phase space projection in {dir_label} direction',
                  fontsize=24
                  )
        plt.xlabel(f'{dir_label}', fontsize=20)
        plt.ylabel(f'v_{dir_label}', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if output_path is not None:
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def animate(
            self,
            output_path=None,
            write_fps=30,
            write_bitrate=1800,
            interval=50,
            xlim=None,
            ylim=None,
            zlim=None
            ):
        """
        Animate the motion of the particles in the trap.

        Args:
        output_path: str or None, if str then provides path to save video file
        write_fps: int, fps at which to save video
        write_bitrate: int
        interval: int
        *lim: pair of floats, limits for the axes of the animation
        """
        # Get data for frames
        data = [np.array(self.get_rs(t, live=False)).transpose()
                for t
                in range(self._sample_points)
                ]
        data = np.array(data)
        # Initialise figure and start position
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Set axes
        if xlim is not None:
            ax.set(xlim=xlim)
        if ylim is not None:
            ax.set(ylim=ylim)
        if zlim is not None:
            ax.set(zlim=zlim)
        scatter = [ax.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])]
        
        # Function to update animation
        def update(frame, scatter):
            scatter._offsets3d = data[frame, :, :]
            return scatter
        # Animate
        ani = animation.FuncAnimation(
                fig,
                update,
                self._sample_points,
                fargs=(scatter),
                interval=interval,
                blit=False
                )
        # Display or save
        if output_path is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=write_fps, bitrate=write_bitrate)
            ani.save(output_path, writer=writer)
        else:
            plt.show()
        plt.close()
