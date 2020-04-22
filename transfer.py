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
    I = 100 * consts.u_B
    I = parameter.RampParameter([I], [100E-3, 200E-3, 100, 200])
    u_axis = 10E-3
    y_0 = 3E-3
    y_1 = 0
    y = parameter.SigmoidParameter(y_0, y_1, 100E-3, 200E-3, 50)
    z_0 = 1E-3
    z_1 = 0.1E-3
    z = parameter.SigmoidParameter(z_0, z_1, 300E-3, 400E-3, 50)
    u_wire = wire.UWire(I, u_axis)
    u_trap = trap.ClusterTrapStatic(u_wire)

    # Initialise QP
    b_1_0 = consts.u_B * .6
    b_1 = parameter.RampParameter([b_1_0], [-1, 0, 100E-3, 200E-3])
    qp = field.QuadrupoleField(b_1, r_0=[0, 0, z_0])
    qp_trap = trap.FieldTrap(qp.field)

    # Combination trap
    center = parameter.ArrayParameter([0, y, z])
    combi_trap = trap.SuperimposeTrapWBias([qp_trap, u_trap], center)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = 5E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)

    # Particle loss
    limit = 100E-3
    loss_event = events.OutOfRangeBox(limit, center=[0, 0, z_0])

    # Simulation
    POINTS = 200
    t_0 = 0
    t_end = 500E-3
    dt = 1E-6
    PARTICLES = 3
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
