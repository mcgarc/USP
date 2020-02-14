from USP import *
import time
import numpy as np


def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    height = parameter.SigmoidParameter(0, 1, 9, 11, 2)
    qp = field.QuadrupoleFieldTranslate(consts.u_B * 0.6, height, direction=0)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 0.8E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
 
    # Simulation
    t_end = 2#5
    sim = simulation.Simulation(qp_trap, 0, t_end, 1E-3)
    sim.init_particles(200, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    #sim.plot_start_end_positions()
    sim.plot_temperatures(output_path='results/test/qp_temps.png')
    sim.plot_center(direction=0, output_path='results/test/qp_center_x.png')
    sim.plot_center(direction=1, output_path='results/test/qp_center_y.png')
    sim.plot_center(direction=2, output_path='results/test/qp_center_z.png')
    sim.plot_width(direction=0, output_path='results/test/qp_width_x.png')
    sim.plot_width(direction=1, output_path='results/test/qp_width_y.png')
    sim.plot_width(direction=2, output_path='results/test/qp_width_z.png')
    sim.plot_cloud_volume(output_path='results/test/qp_volume.png', N_points=200)
    sim.plot_cloud_phase_space_volume(output_path='results/test/qp_ps_volum.png', N_points=200)
    yzlim = (-5E-3, 5E-3)
    xlim = (-1E-3, 1.121)
    sim.animate(2000, xlim=yzlim, ylim=yzlim, zlim=yzlim)#, output_path='results/qp_t_100.mp4')

if __name__ == '__main__':
    main()
