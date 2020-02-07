from USP import *
import time

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    qp = field.QuadrupoleField(1)
    qp_trap = trap.FieldTrap(qp.field)

    # Simulation
    t_end = 10
    sim = simulation.Simulation(qp_trap, 0, t_end, 5E-3)
    sim.init_particles(100, 1, 1E-1, 1E-1)
    start = time.time()
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    sim.plot_start_end_positions()
    lim = (-0.3, 0.3)
    sim.animate(1000, xlim=lim, ylim=lim, zlim=lim)#, output_path='results/qp.mp4')

if __name__ == '__main__':
    main()
