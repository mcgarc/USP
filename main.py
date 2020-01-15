from USP import particle
from USP import trap
from USP import wire
from USP import parameter

from USP.utils import grad

import scipy.integrate as integ
import numpy as np
import pathos.multiprocessing as mp


def stepper(mol):
    """
    For use with multiprocessing map. Calls step_integ method of the mol
    argument, so you don't have to
    """
    steps = 1000
    for step in range(steps):
        mol.step_integ()
        # Debuging output
        print(mol.r)

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise particles
    particle_no = 3
    mols = []
    for _ in range(particle_no):
        mol = particle.Particle(
                [0, 0, 0.1 + np.random.normal(0,0.01)],
                [0, 0, 0],
                1
                )
        mols.append(mol)

    # Initialise potential
    cur = parameter.ConstantParameter(1)
    height = parameter.RampParameter([0.1, 0.2], [-2, -1, 50, 1050, float('inf'), float('inf')])
    zcluster = wire.ZWire(cur, 0.1)
    ztrap = trap.ClusterTrap(
            zcluster,
            height
            )

    # Initialise integrators
    t_end = 100000
    max_step = t_end/1000
    for mol in mols:
        mol.init_integ(ztrap.potential, t_end, max_step=max_step)

    # Stepper
    processes = 2
    with mp.ProcessingPool(processes) as p:
        p.map(stepper, mols)


if __name__ == '__main__':
    main()
