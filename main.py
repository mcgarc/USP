import particle
import trap
import wire
import parameter

from utils import grad

import scipy.integrate as integ
import numpy as np

    

def main():
    """
    Test simulation of a single particle in a rising potential
    """
    mol = particle.Particle(
            [0, 0, 0.11],
            [0, 0, 0],
            1
            )
    cur = parameter.ConstantParameter(1)
    #height = parameter.ConstantParameter(0.1)
    height = parameter.RampParameter([0.1, 0.2], [-2, -1, 50, 1050, float('inf'), float('inf')])
    zcluster = wire.ZWire(cur, 0.1)
    ztrap = trap.ClusterTrap(
            zcluster,
            height
            )
    t_end = 100000
    max_step = t_end/1000
    mol.init_integ(ztrap.potential, t_end, max_step=max_step)
    while mol.t < t_end:
        mol.step_integ()
        print('{}: {}'.format(mol.t, mol.r))


if __name__ == '__main__':
    main()
