import particle
import trap
import wire
import parameter

from utils import grad

import scipy.integrate as integ
import numpy as np

    

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
    for step in range(1000):
        for mol in mols:
            mol.step_integ()
        positions = np.array([ mol.r for mol in mols ])
        times = [ mol.t for mol in mols ] 
        print('{}: {}'.format(times, positions.mean(axis=0)))


if __name__ == '__main__':
    main()
