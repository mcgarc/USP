from USP import *
import time
import numpy as np
import os

def main():
    """
    Test simulation of particles transferring between traps
    """

    # Save script file with results
    with open(__file__, 'r') as f:
        copy_script_path = f'runscipt.py'
        with open(copy_script_path, 'w') as f_out:
            for line in f:
                f_out.write(line)
            f_out.close()
        f.close()

    # Initialise U wire
    # Thick wire should be able to take 100A
    I = 10 * consts.u_B
    I = parameter.RampParameter([I], [2E-3, 5E-3, 1, 1.1])
    z_0 = 1E-3
    z_1 = 1E-6
    h = parameter.SigmoidParameter(z_0, z_1, 8E-3, 12E-3, 1E-3)
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

    # Simulation
    POINTS = 400
    t_0 = 0
    t_end = 15E-3
    dt = 1E-6
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

    sim.pickle(prepend_path='results/')


if __name__ == '__main__':
    main()
