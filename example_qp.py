from USP import *
import time

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    qp = field.QuadrupoleField(100)
    qp_trap = trap.FieldTrap(qp.field)

    # Simulation
    sim = simulation.Simulation(4, qp_trap, 0, 10, 1E-2)
    sim.init_particles(2, 1, 1, 1)
    sim.run()

    quit()

    # Visualisation
    sim.plot_start_end_positions()

if __name__ == '__main__':
    main()
