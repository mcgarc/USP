import particle
import trap
import wire
import parameter

from utils import grad

import scipy.integrate as integ
import numpy as np


def Q_dot(t, Q, potential):
    """
    Return the time derivative of Q, the 6-vector position in momentum space,
    for a particle in given potential at time t. (Hamilton's equations)
    """
    r = Q[:3]
    v = Q[3:]
    dvx_dt = -grad(potential, t, r, 'x')
    dvy_dt = -grad(potential, t, r, 'y')
    dvz_dt = -grad(potential, t, r, 'z')
    dQ_dt = np.array([
            v[0],
            v[1],
            v[2],
            dvx_dt,
            dvy_dt,
            dvz_dt
            ])
    return dQ_dt
    

def main():
    mol = particle.Particle(
            [0, 0, 0.11],
            [0, 0, 0],
            1E-20,
            )
    cur = parameter.ConstantParameter(1)
    #height = parameter.ConstantParameter(0.1)
    height = parameter.RampParameter([0.1, 0.2], [-2, -1, 50, 1050, float('inf'), float('inf')])
    zcluster = wire.ZWire(cur, 0.1)
    ztrap = trap.ClusterTrap(
            zcluster,
            height
            )
    # RK45 takes a function with args t and Q, so pass in potential here
    Q_dot_wpot = lambda t, Q: Q_dot(t, Q, ztrap.potential)
    t_end = 100000
    max_step = t_end/1000
    ans = integ.RK45(Q_dot_wpot, 0, mol.Q, t_end, max_step=max_step)
    while ans.t < t_end:
        ans.step()
        print(ans.y[:3])


if __name__ == '__main__':
    main()
