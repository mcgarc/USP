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
import numpy as np
from itertools import repeat
import multiprocessing as mp
import pickle
import time

from matplotlib import pyplot as plt
import matplotlib.animation as animation


def _integ(particle, potential, t_0, t_end, dt, out_times):
    """
    Call init_integ of a passed mol. For parallelisation
    """
    particle.integ(potential, t_0, t_end, dt)
    return particle


class Simulation:
    """
    """

    def __init__(self,
            process_no,
            trap,
            t_0,
            t_end,
            dt,
            ):
        """
        """
        self._process_no = process_no
        self.set_trap(trap)
        self._t_0 = t_0
        self._t_end = t_end
        self._dt = dt
        self._eval_times = np.arange(t_0, t_end, dt)
        self._particles = None
        self.run_time = None

    @property
    def particles(self):
        return self._particles

    @property
    def result(self):
        """
        TODO: May be helpful to provide some sort of reduced form of output for
        exporting
        """
        raise NotImplemented

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
            raise ValueError(
            'trap is not of valid type, expected None or AbstractTrap'
            )

    def get_rs(self, t):
        """
        Return a list of all particles positions at the given time
        """
        rs = [ p.r(t) for p in self._particles ]
        return rs

    def get_vs(self, t):
        """
        Return a list of all particles velocities at the given time
        """
        vs = [ p.v(t) for p in self._particles ]
        return vs

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
        Run the simulation using multiprocessing. Creates particles
        """
        # Check that a trap has been specified
        if self._trap is None:
            raise RuntimeError('No trap has been specified for this simulation')

        # Create args for _integ running in parallel
        args = zip(
                self._particles,
                repeat(self._trap.potential),
                repeat(self._t_0),
                repeat(self._t_end),
                repeat(self._dt),
                repeat([0, self._t_end]) # FIXME Should be arbitrary
                )

        # Multiprocessing
        start_time = time.time()
        with mp.Pool(self._process_no) as p:
            self._particles = p.starmap(_integ, args)
        self.run_time = time.time() - start_time

    def init_particles(
            self,
            particle_no,
            mass,
            r_sigma,
            v_sigma,
            r_centre = [0, 0, 0],
            v_centre = [0, 0, 0],
            seed = None
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

        if seed is not None:
            np.seed(seed)

        # Clean sigma input into np arrays of length 3
        if type(r_sigma) in [float, int]:
            r_sigma = [r_sigma, r_sigma, r_sigma]
        if type(v_sigma) in [float, int]:
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
              mass
              )
            for i in range(particle_no)
            ]

    def get_total_energy(self, t):
        """
        Get the total energy (KE + potential) at a specified time 
        """
        energies = [p.energy(t, self._trap.potential) for p in self._particles]
        return np.sum(np.array(energies))

    def plot_start_end_positions(self):
        """
        """
        start_rs = np.array(self.get_rs(self._t_0)).transpose()
        end_rs = np.array(self.get_rs(self._t_end)).transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(start_rs[0], start_rs[1], start_rs[2], 'b')
        ax.scatter(end_rs[0], end_rs[1], end_rs[2], 'r')
        plt.show()

    def plot_phase_diagram(
            self,
            particle_index,
            dir_index,
            N_points,
            output_path = None,
            time_gradient=False,
            colorbar=False,
            figsize=(6,4),
            dpi=300,
            **plot_kwargs
            ):
        """
        Plot the phase space diagram of a particle in the trap over time.

        Args:
        particle_index: int, the paticle whose PSD is to be shown
        dire_index: int, corresponds to the x, y or z direction
        N_points: int, no. of points to plot
        output_path: str or None, if None then show graph, otherwise save it
        time_gradient: bool, if true then shade the points for time
        colorbar: bool, if time_gradient is true then colorbar will create a
            colour legend for the times
        figsize: pair of ints, size of output plot
        dpi:int, dpi of output plot
        plot_kwargs: kwargs to be passed to the scatter plot fn
        """
        # Get coordinates
        times = np.linspace(self._t_0, self._t_end, N_points)
        particle = self._particles[particle_index]
        Q_projections = [particle.Q_projection(t, 0) for t in times]
        Q_projections = np.array(Q_projections).transpose()
        # Setup time gradient if required
        if time_gradient:
            plot_kwargs['c'] = times
        # Create plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.05,0.05,0.8,0.8])
        cm = plt.get_cmap('winter')
        sc = ax.scatter(Q_projections[0], Q_projections[1], cmap=cm, **plot_kwargs)
        # Colorbar
        if time_gradient and colorbar:
            cbar = plt.colorbar(sc)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('time', fontsize=20)
        # Label axes
        dir_label = {0:'x', 1:'y', 2:'z'}[dir_index]
        plt.title(f'Phase space projection in {dir_label} direction', fontsize=24)
        plt.xlabel(f'{dir_label}', fontsize=20)
        plt.ylabel(f'v_{dir_label}', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if output_path is not None:
            fig.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()

    def animate(
            self,
            N_frames,
            output_path = None,
            write_fps = 30,
            write_bitrate = 1800,
            interval = 50
            ):
        """
        Animate the motion of the particles in the trap.

        Args:
        N_frames: int, number of frames to animate
        output_path: str or None, if str then provides path to save video file
        write_fps: int, fps at which to save video
        write_bitrate: int
        interval: int
        """
        # Get data for frames
        frame_times = np.linspace(self._t_0, self._t_end, N_frames)
        data = [np.array(self.get_rs(t)).transpose() for t in frame_times]
        data = np.array(data)
        # Initialise figure and start position
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlim=(-0.2, 100))
        scatter = [ax.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])]
        # Function to update animation
        def update(frame, scatter):
            scatter._offsets3d = data[frame, :, :]
            return scatter
        # Animate
        ani = animation.FuncAnimation(
                fig,
                update,
                N_frames,
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
