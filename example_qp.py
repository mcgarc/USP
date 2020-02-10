from USP import *
import numpy as np

def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    qp = field.QuadrupoleField(consts.u_B * 0.6)
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
    lim = (-1E5, 1E5)
    sim.animate(1000)#, output_path='results/qp.mp4')

if __name__ == '__main__':
    main()
