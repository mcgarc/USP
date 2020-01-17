from USP import *

import scipy.integrate as integ
import numpy as np
import multiprocessing as mp
from itertools import repeat
import time

np.random.seed(11111)

def integ(mol, potential, t_end, max_step, out_times):
    """
    Call init_integ of a passed mol. For parallelisation
    """
    mol.integ(potential, t_end, max_step=max_step)
    #print(mol.index) # Useful for debugging
    # Check conservation of energy
    #print (np.dot(mol.v(0), mol.v(0)) - np.dot(mol.v(t_end), mol.v(t_end)))
    # Output molecule position at specified times
    return [mol.Q(t) for t in out_times]

def qp_simulation(
        particle_no,
        r_sigma,
        v_sigma,
        mass,
        qp_b1,
        t_end,
        max_step,
        ):
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise particles
    mols = [
        particle.Particle(
          [
              np.random.normal(0, r_sigma),
              np.random.normal(0, r_sigma),
              np.random.normal(0, r_sigma)
          ],
          [
              np.random.normal(0, v_sigma),
              np.random.normal(0, v_sigma),
              np.random.normal(0, v_sigma)
          ],
          mass
          )
        for _ in range(particle_no)
        ]

    # Initialise QP
    qp = field.QuadrupoleField(qp_b1)
    qp_trap = trap.FieldTrap(qp.field)

    # Output times
    times = np.linspace(0, t_end, 20)

    # Perform integration
    processes = 4
    args = zip(
            mols,
            repeat(qp_trap.potential),
            repeat(t_end),
            repeat(max_step),
            repeat(times)
            )
    with mp.Pool(processes) as p:
        mol_Qs = p.starmap(integ, args)

    utils.create_output_csv(times, mol_Qs)

def main():
    # Params
    particle_no = 1000
    r_sigma = 1E-1
    v_sigma = 1E-4
    mass = 1E-4
    qp_b1 = 100
    t_end = 1E2
    max_step = np.inf

    # Time and run simulation
    start = time.time()
    qp_simulation(
        particle_no,
        r_sigma,
        v_sigma,
        mass,
        qp_b1,
        t_end,
        max_step,
        )
    run_time = time.time() - start
    print(f'Simulation runtime: {run_time:.2f}s')

    # Graphical output
    lims = (-3E-1, 3E-1)
    lims = None
    plot.plot_positions(
            ['output/0.0.csv', f'output/{t_end}.csv'],
            ['b', 'r'],
            x_lim=lims,
            y_lim=lims,
            z_lim=lims
            )


if __name__ == '__main__':
    main()
