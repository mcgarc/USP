from USP import *
import numpy as np
import time

def harmonic_potential(t, r):
    V_0 = 0
    m = 1
    omega_0 = 1
    return V_0 + 0.5 * m * omega_0 * np.dot(r, r)

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise harmonic trap
    h_trap = trap.AbstractTrap()
    h_trap.potential = harmonic_potential

    # Simulation
    t_end = 30
    sim = simulation.Simulation(4, h_trap, 0, t_end, 1E-2)
    sim.init_particles(100, 1, 1, 1)
    sim.run()
    print(f'runtime: {sim.run_time}')

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    # Visualisation
    #sim.plot_start_end_positions()

#    sim.plot_phase_diagram(0, 0, 100, time_gradient=True, colorbar=True,
#            output_path='results/phasespace_x')

    #sim.animate(1000)

if __name__ == '__main__':
    main()
