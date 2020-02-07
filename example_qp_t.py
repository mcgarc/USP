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
        return np.array([0, 0, t/2.])

    def field(self, t, r):
        r = np.array(r) - self.r_0 + self.D_r(t)
        scale = np.array([-0.5, -0.5, 1.])
        return self.b_1 * scale * r

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    qp = QuadrupoleFieldMoving(10)
    qp_trap = trap.FieldTrap(qp.field)

    # Simulation
    t_end = 10
    sim = simulation.Simulation(qp_trap, 0, t_end, 5E-2)
    sim.init_particles(50, 1, 1E-1, 1E-1)
    start = time.time()
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    sim.plot_start_end_positions()
    lim = (-3, 3)
    sim.animate(1000, xlim=lim, ylim=lim, zlim=lim)#, output_path='results/qp.mp4')

if __name__ == '__main__':
    main()
