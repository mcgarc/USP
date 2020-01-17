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
    sim = simulation.Simulation(4, qp_trap, 1E2, 20)
    sim.init_particles(1000, 1E-4, 1E-1, 1E-4)
    sim.run()

    # Visualisation
    sim.plot_start_end_positions()

if __name__ == '__main__':
    main()
