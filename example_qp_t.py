from USP import *
import time
import numpy as np

class QuadrupoleFieldMoving(field.QuadrupoleField):
    """
    Adapt quadrupole field to move in time
    """

    def D_r(self, t):
        """
        Change in r over time
        """
        return np.array([0, 0, t/10.])

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
    t_end = 1
    sim = simulation.Simulation(qp_trap, 0, t_end, 1E-3)
    sim.init_particles(50, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    sim.plot_start_end_positions()
    sim.plot_temperatures()
    xylim = (-1E-2, 1E-2)
    zlim = (-1E-3, 0.101)
    sim.animate(1000, xlim=xylim, ylim=xylim, zlim=zlim)#, output_path='results/qp.mp4')

if __name__ == '__main__':
    main()
