from USP import *
import time
import numpy as np


def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    height = parameter.SigmoidParameter(0, 1, 9, 11, 2)
    qp = field.QuadrupoleFieldTranslate(consts.u_B * 0.6, height)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 1E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
 
    # Simulation
    t_end = 25
    sim = simulation.Simulation(qp_trap, 0, t_end, 1E-3)
    sim.init_particles(10, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    #sim.plot_start_end_positions()
    sim.plot_temperatures(output_path='results/test/qp_temps.png')
    sim.plot_center(output_path='results/test/qp_center.png')
    sim.plot_width(direction=0, output_path='results/test/qp_width_x.png')
    sim.plot_width(direction=1, output_path='results/test/qp_width_y.png')
    sim.plot_width(direction=2, output_path='results/test/qp_width_z.png')
    xylim = (-1E-2, 1E-2)
    zlim = (-1E-3, 1.121)
    #sim.animate(2000, xlim=xylim, ylim=xylim, zlim=zlim)#, output_path='results/qp_t_100.mp4')

if __name__ == '__main__':
    main()
