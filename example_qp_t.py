from USP import *
import time
import numpy as np

class QuadrupoleFieldMoving(field.QuadrupoleField):
    """
    Adapt quadrupole field to move in time
    """

    def D_r(self, t, t_start=4, t_end=14, r_shift=0.1, v=1):
        """
        Change in r over time
        """
        t_start = float(t_start)
        t_end = float(t_end)
        r_shift = float(r_shift)
        if t < t_start:
            return np.array([0, 0, 0])
        elif t > t_end:
            return np.array([0, 0, r_shift])
        else:
            denom = 1 + np.exp(-v * (t - (t_start + t_end)/2))
            r = r_shift / denom
            return np.array([0, 0, r])

    def field(self, t, r):
        r = np.array(r) - self.r_0 - self.D_r(t)
        scale = np.array([-0.5, -0.5, 1.])
        return self.b_1 * scale * r

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    qp = QuadrupoleFieldMoving(consts.u_B * 0.6)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 1E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
 
    # Simulation
    t_end = 20
    sim = simulation.Simulation(qp_trap, 0, t_end, 1E-3)
    sim.init_particles(50, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    sim.plot_start_end_positions()
    sim.plot_temperatures()
    sim.plot_center()
    xylim = (-1E-2, 1E-2)
    zlim = (-1E-3, 0.121)
    #sim.animate(2000, xlim=xylim, ylim=xylim, zlim=zlim, output_path='results/qp_move.mp4')

if __name__ == '__main__':
    main()
