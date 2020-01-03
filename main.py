import particle
import trap
import wire
import current

from utils import grad

import scipy.integrate as integ
import numpy as np


def fun(t, Q, potential):
    """
    Q = (r, v) is the 6-vector of position in phase space
    """
    r = Q[:3].flatten()
    v = Q[3:].flatten()
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
    cur = current.ConstantCurrent(1)
    zcluster = wire.ZWire(cur, 0.1)
    ztrap = trap.ClusterTrapStaticTR(
            zcluster,
            0,
            [0, 0, 0.1]
            )
    fun2 = lambda t, Q: fun(t, Q, ztrap.potential)
    t_end = 100000
    max_step = t_end/1000
    ans = integ.RK45(fun2, 0, mol.Q, t_end, vectorized=True, max_step=max_step)
    while ans.t < t_end:
        ans.step()
        print(ans.y[:3])


if __name__ == '__main__':
    main()

