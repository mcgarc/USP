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
        copy_script_path = f'runscript.py'
        with open(copy_script_path, 'w') as f_out:
            for line in f:
                f_out.write(line)
            f_out.close()
        f.close()

    # Initialise U wire
    # Thick wire should be able to take 100A
    I_u1_0 = 100 * consts.u_B
    I_u1_1 = 100 * consts.u_B
    I_u1 = parameter.RampParameter([I_u1_0, I_u1_1], [100E-3, 200E-3, 300E-3, 500E-3, 600E-3, 700E-3])
    u_axis = 10E-3
    y_0 = 2E-3
    y_1 = 0
    y = parameter.SigmoidParameter(y_0, y_1, 120E-3, 180E-3, 50)
    z_0 = 3E-3
    z_1 = .5E-3
    z = parameter.SigmoidParameter(z_0, z_1, 320E-3, 480E-3, 50)
    u_wire = wire.UWire(I_u1, u_axis)
    u_trap = trap.ClusterTrapStatic(u_wire)
    
    # Initialise QP
    b_1_0 = consts.u_B * .6
    b_1 = parameter.RampParameter([b_1_0], [-1, 0, 100E-3, 200E-3])
    qp = field.QuadrupoleField(b_1, r_0=[0, y_0, z_0])
    qp_trap = trap.FieldTrap(qp.field)
    
    # Initialise Z wire
    z_axis = 3E-3 # 3mm as per chip design
    I_z1_0 = 1 * consts.u_B
    I_z1 = parameter.RampParameter([I_z1_0], [600E-3, 700E-3, 800E-3, 1000E-3])
    z_wire_1 = wire.ZWire(I_z1, z_axis)
    z_trap_1 = trap.ClusterTrapStatic(z_wire_1)
    
    # Combination trap
    center = parameter.ArrayParameter([0, y, z])
    combi_trap = trap.SuperimposeTrapWBias([qp_trap, u_trap, z_trap_1], center)

    # Initial conditions
    T = 50E-6
    mass = consts.m_Rb
    r_spread = .5E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)

    # Particle loss
    limit = 2E-3
    loss_event = events.OutOfRangeSphere(limit, center)

    # Particle generators
    r_gen = rand.NormalGenerator([0, y_0, z_0], 3 * [r_spread])
    v_gen = rand.NormalGenerator(0, v_spread, length=3)

    # Simulation
    POINTS = 200
    t_0 = 0
    t_end = 50E-3
    dt = 1E-6
    PARTICLES = 100
    sim = simulation.Simulation(
            combi_trap,
            t_0,
            t_end,
            dt,
            POINTS,
            events=loss_event,
            process_no=48
            )
    sim.init_particles(PARTICLES, mass, r_gen, v_gen)

    sim.run()

    sim.pickle(prepend_path='results/')


if __name__ == '__main__':
    main()
