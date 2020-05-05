from USP import *
import time
import numpy as np
import os

z_0 = 3E-3
z_1 = .5E-3
y_0 = 2E-3
y_1 = 0

def u_trap():
    """
    U trap as we hold after coild transfer
    """
    I_0 = 100 * consts.u_B
    I = parameter.ConstantParameter(I_0)
    u_axis = 10E-3
    z = parameter.ConstantParameter(z_0)
    u_wire = wire.UWire(I, u_axis)
    u_trap = trap.ClusterTrap(u_wire, z)
    return u_trap
    

def main():
    """
    Test simulation of particles transferring between traps
    """
    
    # Trap
    trap = u_trap()

    # Initial conditions
    T = 600E-6
    mass = consts.m_CaF
    r_spread = 5E-3
    v_spread = np.sqrt(2*consts.k_B*T/mass)
    
    # Particle generators
    r_gen = rand.UniformGenerator([-10E-3, -10E-3, 1E-9], [10E-3, 10E-3, 20E-3])
    v_gen = rand.UniformGenerator(-v_spread, v_spread, 3)

    # Particle loss
    limit = 5E-3
    center = parameter.ArrayParameter([0, 0, z_0])
    loss_event = events.OutOfRangeSphere(limit, center=center)

    # Simulation
    POINTS = 200
    t_0 = 0
    t_end = 200E-3
    dt = 1E-6
    PARTICLES = 10
    sim = simulation.Simulation(
            trap,
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
