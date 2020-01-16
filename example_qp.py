from USP import particle
from USP import trap
from USP import wire
from USP import parameter
from USP import field
from USP import utils

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

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise particles
    particle_no = 100
    mols = []
    for _ in range(particle_no):
        mol = particle.Particle(
                [
                    np.random.normal(0, 0.21),
                    np.random.normal(0, 0.21),
                    np.random.normal(0, 0.21)
                ],
                [
                    np.random.normal(0, 0.21),
                    np.random.normal(0, 0.21),
                    np.random.normal(0, 0.21)
                ],
                1
                )
        mols.append(mol)

    # Initialise QP
    qp = field.QuadrupoleField(10)
    qp_trap = trap.FieldTrap(qp.field)

    # Output times
    t_end = 1E2
    times = np.linspace(0, t_end, 20)

    # Perform integration
    max_step = t_end / 10
    processes = 4
    args = zip(mols, repeat(qp_trap.potential), repeat(t_end), repeat(max_step), repeat(times))
    with mp.Pool(processes) as p:
        mol_Qs = p.starmap(integ, args)
        for Qs in mol_Qs:
            for Q in Qs:
                print(Q)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)

