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
    t_0 = 0
    t_end = 1E-3
    dt = 0.5E-6
    PARTICLES = 4
    r_centre = [0., 0., z_0]
    sim = simulation.Simulation(
            combi_trap,
            t_0,
            t_end,
            dt,
            POINTS,
            events=loss_event,
            process_no=48
            )
    sim.init_particles(PARTICLES, mass, r_spread, v_spread, r_centre=r_centre)

    sim.run()

    sim.pickle('results/output.pickle')


if __name__ == '__main__':
    main()
