from USP import *
import time
import numpy as np
import os

def main():
    """
    Test simulation of a few particles in a rising potential
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

    # Initialise z wire
    I = parameter.ConstantParameter(consts.u_B * 1)
    z_0 = 1E-3
    h = parameter.ConstantParameter(z_0)
    z_wire = wire.ZWire(I, z_0)
    z_trap = trap.ClusterTrap(z_wire, h)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    z_fact = 1
    r_spread_s = (z_0/10) / (2 + z_fact)
    r_spread = [r_spread_s, r_spread_s, r_spread_s*z_fact]
    v_spread = np.sqrt(2*consts.k_B*T/mass)

    # Particle loss (translate with the stage)
    loss_rad = 10 * z_0
    loss_rad = 0.8
    loss_event = events.OutOfRangeSphere(loss_rad)
 
    # Simulation
    POINTS = 300
    t_end = 3
    PARTICLES = 10
    r_centre = [0., 0., z_0]
    sim = simulation.Simulation(
            z_trap,
            0,
            t_end,
            1E-3,
            POINTS,
            events=loss_event
            )
    sim.init_particles(PARTICLES, mass, r_spread, v_spread, r_centre=r_centre)

    sim.run()

    E_0 = sim.get_total_energy(0)
    E_end = sim.get_total_energy(-1)
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
    sim.plot_position_histogram(0, 0, output_path=f'{path}r_t0_hist_x.png')
    sim.plot_position_histogram(0, 1, output_path=f'{path}r_t0_hist_y.png')
    sim.plot_position_histogram(0, 2, output_path=f'{path}r_t0_hist_z.png')
    sim.plot_position_histogram(-1, 0, output_path=f'{path}r_t1_hist_x.png')
    sim.plot_position_histogram(-1, 1, output_path=f'{path}r_t1_hist_y.png')
    sim.plot_position_histogram(-1, 2, output_path=f'{path}r_t1_hist_z.png')
    sim.plot_momentum_histogram(0, 0, output_path=f'{path}p_t0_hist_x.png')
    sim.plot_momentum_histogram(0, 1, output_path=f'{path}p_t0_hist_y.png')
    sim.plot_momentum_histogram(0, 2, output_path=f'{path}p_t0_hist_z.png')
    sim.plot_momentum_histogram(-1, 0, output_path=f'{path}p_t1_hist_x.png')
    sim.plot_momentum_histogram(-1, 1, output_path=f'{path}p_t1_hist_y.png')
    sim.plot_momentum_histogram(-1, 2, output_path=f'{path}p_t1_hist_z.png')
    xlim = (-1E-1, 1E-1)
    ylim = (-10E-3, 10E-3)
    zlim = (ylim[0], ylim[1] + z_0)
    sim.animate(xlim=xlim, ylim=ylim, zlim=zlim, output_path=f'{path}anim.mp4')


if __name__ == '__main__':
    main()
