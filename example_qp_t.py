from USP import *
import time
import numpy as np


def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise QP
    height = parameter.SigmoidParameter(0, 1, 9, 11, 2)
    height = parameter.ConstantParameter(0)
    qp = field.QuadrupoleFieldTranslate(consts.u_B * 0.6, height, direction=0)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 0.8E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
 
    # Simulation
    t_end = 10
    PARTICLES = 1E1
    sim = simulation.Simulation(qp_trap, 0, t_end, 5E-4)
    sim.init_particles(PARTICLES, mass, r_spread, v_spread)
    sim.run()

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

#    sim.save_to_pickle('results/staticqp.pickle')

    #sim.plot_start_end_positions()
    POINTS = 250
    PATH = 'results/test/'
    sim.plot_temperatures(output_path=f'{PATH}temps.png', N_points=POINTS)
    sim.plot_center(direction=0, output_path=f'{PATH}center_x.png', N_points=POINTS)
    sim.plot_center(direction=1, output_path=f'{PATH}center_y.png', N_points=POINTS)
    sim.plot_center(direction=2, output_path=f'{PATH}center_z.png', N_points=POINTS)
    sim.plot_width(direction=0, output_path=f'{PATH}width_x.png', N_points=POINTS)
    sim.plot_width(direction=1, output_path=f'{PATH}width_y.png', N_points=POINTS)
    sim.plot_width(direction=2, output_path=f'{PATH}width_z.png', N_points=POINTS)
    sim.plot_cloud_volume(output_path=f'{PATH}volume.png', N_points=POINTS)
    sim.plot_cloud_phase_space_volume(output_path=f'{PATH}ps_volum.png', N_points=POINTS)
    sim.plot_position_histogram(0, direction=0, output_path=f'{PATH}r_t0_hist_x.png')
    sim.plot_position_histogram(0, direction=1, output_path=f'{PATH}r_t0_hist_y.png')
    sim.plot_position_histogram(0, direction=2, output_path=f'{PATH}r_t0_hist_z.png')
    sim.plot_position_histogram(t_end, direction=0, output_path=f'{PATH}r_t1_hist_x.png')
    sim.plot_position_histogram(t_end, direction=1, output_path=f'{PATH}r_t1_hist_y.png')
    sim.plot_position_histogram(t_end, direction=2, output_path=f'{PATH}r_t1_hist_z.png')
    sim.plot_momentum_histogram(0, direction=0, output_path=f'{PATH}p_t0_hist_x.png')
    sim.plot_momentum_histogram(0, direction=1, output_path=f'{PATH}p_t0_hist_y.png')
    sim.plot_momentum_histogram(0, direction=2, output_path=f'{PATH}p_t0_hist_z.png')
    sim.plot_momentum_histogram(t_end, direction=0, output_path=f'{PATH}p_t1_hist_x.png')
    sim.plot_momentum_histogram(t_end, direction=1, output_path=f'{PATH}p_t1_hist_y.png')
    sim.plot_momentum_histogram(t_end, direction=2, output_path=f'{PATH}p_t1_hist_z.png')
    yzlim = (-5E-3, 5E-3)
    xlim = (-1E-3, 1.121)
    #sim.animate(2000, xlim=yzlim, ylim=yzlim, zlim=yzlim)#, output_path='results/qp_t_100.mp4')

if __name__ == '__main__':
    main()
