from USP import *
import numpy as np
import time



def main():
    """
    Test simulation of a few particles in a Zeeman guide
    """

    # Initialise Zeeman guide
    z_guide = trap.ZeemanGuide()

    # Simulation
    t_end = 10E-3
    mass = consts.m_CaF
    hole_diameter = 4E-3
    r_spread = hole_diameter/4
    T = 4
    v_spread = np.sqrt(2*consts.k_B*T/mass)
    v_z = 144
    sim = simulation.Simulation(z_guide, 0, t_end, 1E-6)
    sim.init_particles(30, mass, r_spread, v_spread, v_centre=[0, 0, v_z])
    sim.save_Q_to_csv(0, 'results/start_Q.csv')
    sim.run()
    sim.save_Q_to_csv(t_end, 'results/end_Q.csv')

    # Diagnosis
    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    # Visualisation
    sim.plot_start_end_positions()
    xylim =0.005
    sim.animate(500, xlim=(-xylim, xylim), ylim=(-xylim, xylim), zlim=(-0.2,2))
            #, output_path='results/zeeman.mp4')

if __name__ == '__main__':
    main()
