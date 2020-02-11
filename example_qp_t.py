from USP import *
import time
import numpy as np


def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    height = parameter.SigmoidParameter(0, 1, 4, 14, 1)
    qp = field.QuadrupoleFieldTranslate(consts.u_B * 0.6, height)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 1E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
 
    # Simulation
    t_end = 20
    sim = simulation.Simulation(qp_trap, 0, t_end, 1E-3)
    sim.init_particles(20, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    sim.plot_start_end_positions()
    sim.plot_temperatures()
    sim.plot_center()
    xylim = (-1E-2, 1E-2)
    zlim = (-1E-3, 0.121)
    #sim.animate(2000, xlim=xylim, ylim=xylim, zlim=zlim, output_path='results/qp_move.mp4')

if __name__ == '__main__':
    main()
