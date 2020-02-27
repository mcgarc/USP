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

    # Initialise QP (translate with the stage)
    stage = parameter.SigmoidParameter(0, 1, 3, 5.5, 2)
    qp = field.QuadrupoleFieldTranslate(consts.u_B * 0.6, stage, direction=0)
    qp_trap = trap.FieldTrap(qp.field)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    z_fact = 0.5
    r_spread_s =2.4E-3 / (2 + z_fact)
    r_spread = [r_spread_s, r_spread_s, r_spread_s*z_fact]
    v_spread = np.sqrt(2*consts.k_B*T/mass)

    # Particle loss (translate with the stage)
    loss_event = events.OutOfRangeSphereTranslate(100E-3, stage)
 
    # Simulation
    POINTS = 500
    t_end = 6.5
    PARTICLES = 50
    sim = simulation.Simulation(
            qp_trap,
            0,
            t_end,
            1E-3,
            POINTS,
            events=loss_event
            )
    sim.init_particles(PARTICLES, mass, r_spread, v_spread)

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
    xlim = (-1E-3, 1.121)
    yzlim = (-5E-3, 5E-3)
    sim.animate(xlim=yzlim, ylim=yzlim, zlim=yzlim, output_path=f'{path}anim.mp4')


if __name__ == '__main__':
    main()
