from USP import *
import numpy as np
import time

def zeeman_potential(t, r):
    V_0 = 0
    m = 1
    omega_0 = 1
    return V_0 + 0.5 * m * omega_0 * (r[1]*r[1] + r[2]*r[2])

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise harmonic trap
    h_trap = trap.AbstractPotentialTrap(zeeman_potential)

    # Simulation
    t_end = 150
    sim = simulation.Simulation(4, h_trap, 0, t_end, 1E-2)
    sim.init_particles(80, 1, 0.1, 0.1, v_centre=[1, 0, 0])
    sim.run()
    print(f'runtime: {sim.run_time}')

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    # Visualisation
    sim.plot_start_end_positions()

#    sim.plot_phase_diagram(0, 0, 100, time_gradient=True, colorbar=True,
#            output_path='results/phasespace_x')

    sim.animate(1000, output_path='results/zeeman.mp4')

if __name__ == '__main__':
    main()
