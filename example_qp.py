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
    sim = simulation.Simulation(4, qp_trap, 0, t_end, 1E-2)
    sim.init_particles(100, 1, 1E-1, 1E-3)
    start = time.time()
    sim.run()
    print('runtime: {}'.format(time.time() - start))

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    # Visualisation
    #sim.plot_start_end_positions()
    sim.animate()

if __name__ == '__main__':
    main()
