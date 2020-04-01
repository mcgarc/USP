from USP import *
import time
import numpy as np
import os

def main():
    """
    Test simulation of particles transferring between traps
    """

    # Setup output directory
    start_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = f'results/{start_time_str}/'
    if not os.path.exists(path):
        os.mkdir(path)

    # Save script file with results
    with open(__file__, 'r') as f:
        copy_script_path = f'{path}runscipt.py'
        with open(copy_script_path, 'w') as f_out:
            for line in f:
                f_out.write(line)
            f_out.close()
        f.close()

    # Initialise U wire
    # Thick wire should be able to take 100A
    I = 10 * consts.u_B
    #I = parameter.RampParameter([consts.u_B * I], [2E-3, 5E-3, 1, 1.1])
    I = parameter.RampParameter([I], [2E-3, 5E-3, 1, 1.1])
    z_0 = 1E-3
    z_1 = 1E-6
    #I = parameter.ConstantParameter(consts.u_B * I)
    #h = parameter.SigmoidParameter(z_0, z_1, 4E-3, 6E-3, 1E-3)
    h = parameter.ConstantParameter(z_0)
    u_wire = wire.UWire(I, z_0)
    u_trap = trap.ClusterTrap(u_wire, h, bias_scale=[0, 1, 1])

    # Initialise QP
    b_1_0 = consts.u_B * .6
    b_1 = parameter.RampParameter([b_1_0], [-1, 0, 2E-3, 5E-3])
    qp = field.QuadrupoleField(b_1, r_0=[0, 0, z_0])
    qp_trap = trap.FieldTrap(qp.field)

    # Combination trap
    combi_trap = trap.SuperimposeTrap([qp_trap, u_trap])

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 0.05E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)

    # Particle loss
    limit = 10E-3
    loss_event = events.OutOfRangeBox(limit, center=[0, 0, z_0])

#    u_trap.plot_potential_cut(
#            1,
#            0.5E-3,
#            2*z_0,
#            'z',
#            cut_point=[0,0,z_0]
#            #output_path=f'{path}_pot_plot_x'
#            )
#    quit()
 
    # Simulation
    POINTS = 400
    #t_end = 7E-3
    t_0 = 0
    t_end = 1E-5
    dt = 0.5E-6
    PARTICLES = 1
    r_centre = [0., 0., z_0]
    sim = simulation.Simulation(
            combi_trap,
            t_0,
            t_end,
            dt,
            POINTS,
            events=loss_event
            )
    sim.init_particles(PARTICLES, mass, r_spread, v_spread, r_centre=r_centre)

    sim.run()

    E_0 = sim.get_total_energy(0)
    E_end = sim.get_total_energy(t_end)
    print(f'Energy differential: {(E_0 - E_end) / E_0:.5f} %')
    
    sim.plot_temperatures(output_path=f'{path}temps.png')
    sim.plot_particle_number(output_path=f'{path}particle_no.png')
    sim.plot_center(0, output_path=f'{path}center_x.png')
    sim.plot_center(1, output_path=f'{path}center_y.png')
    sim.plot_center(2, output_path=f'{path}center_z.png')
    sim.plot_width(0, output_path=f'{path}width_x.png')
    sim.plot_width(1, output_path=f'{path}width_y.png')
    sim.plot_width(2, output_path=f'{path}width_z.png')
    sim.plot_velocity_width(0, output_path=f'{path}velocity_width_x.png')
    sim.plot_velocity_width(1, output_path=f'{path}velocity_width_y.png')
    sim.plot_velocity_width(2, output_path=f'{path}velocity_width_z.png')
    sim.plot_cloud_volume(output_path=f'{path}volume.png')
    sim.plot_cloud_phase_space_volume(output_path=f'{path}ps_volum.png')
    sim.plot_position_histogram(t_0, 0, output_path=f'{path}r_t0_hist_x.png')
    sim.plot_position_histogram(t_0, 1, output_path=f'{path}r_t0_hist_y.png')
    sim.plot_position_histogram(t_0, 2, output_path=f'{path}r_t0_hist_z.png')
    sim.plot_position_histogram(t_end, 0, output_path=f'{path}r_t1_hist_x.png')
    sim.plot_position_histogram(t_end, 1, output_path=f'{path}r_t1_hist_y.png')
    sim.plot_position_histogram(t_end, 2, output_path=f'{path}r_t1_hist_z.png')
    sim.plot_momentum_histogram(t_0, 0, output_path=f'{path}p_t0_hist_x.png')
    sim.plot_momentum_histogram(t_0, 1, output_path=f'{path}p_t0_hist_y.png')
    sim.plot_momentum_histogram(t_0, 2, output_path=f'{path}p_t0_hist_z.png')
    sim.plot_momentum_histogram(t_end, 0, output_path=f'{path}p_t1_hist_x.png')
    sim.plot_momentum_histogram(t_end, 1, output_path=f'{path}p_t1_hist_y.png')
    sim.plot_momentum_histogram(t_end, 2, output_path=f'{path}p_t1_hist_z.png')
    for ind, t in enumerate(np.linspace(t_0, t_end, 5)):
        for d in ['x', 'y', 'z']:
            sim.plot_phase_space_diagram(t, d, f'{path}psd_{d}_{ind}')
    #xlim = (-3*z_0, 3*z_0)
    #ylim = (-3*z_0, 3*z_0)
    #zlim = (xlim[0], xlim[1] + z_0)
    xlim=(-3*r_spread, 3*r_spread)
    ylim = xlim
    zlim = (xlim[0], xlim[1] + z_0)
    sim.animate(xlim=xlim, ylim=ylim, zlim=zlim, output_path=f'{path}anim.mp4')


if __name__ == '__main__':
    main()
