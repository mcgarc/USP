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

def integ(mol, potential, t_end, max_step):
    """
    Call init_integ of a passed mol. For parallelisation
    """
    mol.integ(potential, t_end, max_step=max_step)
    #print(mol.index) # Useful for debugging
    # Check conservation of energy
    #print (np.dot(mol.v(0), mol.v(0)) - np.dot(mol.v(t_end), mol.v(t_end)))

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise particles
    particle_no = 10
    mols = []
    for _ in range(particle_no):
        mol = particle.Particle(
                [
                    np.random.normal(0, 0.01),
                    np.random.normal(0, 0.01),
                    np.random.normal(0, 0.01)
                ],
                [
                    np.random.normal(0, 0.01),
                    np.random.normal(0, 0.01),
                    np.random.normal(0, 0.01)
                ],
                1
                )
        mols.append(mol)

    # Initialise QP
    qp = field.QuadrupoleField(10)
    qp_trap = trap.FieldTrap(qp.field)

    # Perform integration
    t_end = 1E3
    max_step = t_end / 100
    processes = 4
    args = zip(mols, repeat(qp_trap.potential), repeat(t_end), repeat(max_step))
    with mp.Pool(processes) as p:
        p.starmap(integ, args)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)

