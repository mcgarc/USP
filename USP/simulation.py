from USP import trap as USP_trap
from USP import particle
from USP import utils
import numpy as np
from itertools import repeat
import multiprocessing as mp
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _integ(mol, potential, t_end, max_step, out_times):
    """
    Call init_integ of a passed mol. For parallelisation
    """
    mol.integ(potential, t_end, max_step=max_step)
    #print(mol.index) # Useful for debugging
    # Check conservation of energy
    #print (np.dot(mol.v(0), mol.v(0)) - np.dot(mol.v(t_end), mol.v(t_end)))
    # Output molecule position at specified times
    return [mol.Q(t) for t in out_times]


class Simulation:
    """
    """

    def __init__(self,
            process_no = 1,
            trap = None,
            t_end = 1,
            eval_points = 2,
            t_0=0,
            max_step=np.inf
            ):
        """
        """
        self._process_no = process_no
        self.set_trap(trap)
        self._t_end = t_end
        self._eval_times = np.arange(t_0, t_end, eval_points)
        self._max_step = max_step
        self._particles = None
        self._result = None

    @property
    def particles(self):
        return self._particles

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError('Simulation has no results.')
        else:
            return self._result

    def set_trap(self, trap):
        """
        Set trap, checking type in the process. Can also set trap to None.
        """
        if trap is None:
            self._trap = None
        elif isinstance(trap, USP_trap.AbstractTrap):
            self._trap = trap
        else:
            raise ValueError(
            'trap is not of valid type, expected None or AbstractTrap'
            )

    def get_rs(self, time_index):
        """
        Return a list of all particles positions at the given time index
        """
        return self.result[time_index, :, :3].transpose()

    def get_vs(self, time_index):
        """
        Return a list of all particles velocities at the given time index
        """
        return self.result[time_index, :, 3:].transpose()

    def save_result_to_pickle(self, filename):
        """
        Save _result as a pickle
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.result, f)

    def load_result_from_pickle(self, filename):
        """
        Load _result as a picle
        """
        with open(filename, 'rb') as f:
            self._result = pickle.load(f)

    def load_result_from_csv(self, filename):
        """
        Load CSV files into result
        """
        raise NotImplemented

    def save_result_to_csv(self, filename):
        """
        Create a series of CSV files containing Q information for each particle
        at eval times

        Args:
        filename: string, files saved at `output/{filename}_{index}.csv

        Output:
        None, creates CSV files
        """
        for t_index in range(len(self._eval_times)):
            time = times[t_index]
            Qs = self._result[t_index]
            with open(f'output/{filename}_{t_index:03}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter = ' ')
                for Q_index in range(len(Qs)):
                    row = [time, Q_index] + Qs[Q_index]
                    writer.writerow(row)

    def run(self):
        """
        Run the simulation using multiprocessing.

        Output:
        None, populsates self._results TODO: with results object!
        """
        # Check that a trap has been specified
        if self._trap is None:
            raise RuntimeError('No trap has been specified for this simulation')

        # Create args for _integ running in parallel
        args = zip(
                self._particles,
                repeat(self._trap.potential),
                repeat(self._t_end),
                repeat(self._max_step),
                repeat(self._eval_times)
                )

        # Multiprocessing
        with mp.Pool(self._process_no) as p:
            result = p.starmap(_integ, args)

        # result post-processing
        self._result = np.array(result).transpose(1, 0, 2) # TODO Results object

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

    def plot_start_end_positions(self):
        """
        """
        start_rs = self.get_rs(0)
        end_rs = self.get_rs(-1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(start_rs[0], start_rs[1], start_rs[2], 'b')
        ax.scatter(end_rs[0], end_rs[1], end_rs[2], 'r')
        plt.show()
