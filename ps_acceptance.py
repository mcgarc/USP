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
    U trap as we hold after coil transfer
    """
    I_0 = 100 * consts.u_B
    I = parameter.ConstantParameter(I_0)
    u_axis = 10E-3
    z = parameter.ConstantParameter(z_0)
    u_wire = wire.UWire(I, u_axis)
    u_trap = trap.ClusterTrap(u_wire, z)
    return u_trap

def z_trap():
    """
    Z trap as we hold after U transfer
    """
    I_0 = 1 * consts.u_B
    I = parameter.ConstantParameter(I_0)
    z_axis = 3E-3
    z = parameter.ConstantParameter(z_1)
    z_wire = wire.ZWire(I, z_axis)
    z_trap = trap.ClusterTrap(z_wire, z)
    return z_trap
    

def main():
    """
    Test simulation of particles transferring between traps
    """
    
    # Trap
    trap = u_trap()

    # Initial conditions
    T = 6E-3
    mass = consts.m_CaF
    r_spread = 5E-3
    
    # Particle generators
    r_gen = rand.UniformGenerator([-6E-3, -5E-3, 2E-3], [6E-3, 5E-3, 8E-3])
    v_gen = rand.TemperatureGenerator(mass, T, length=3)

    # Particle loss
    limit = 20E-3
    center = parameter.ArrayParameter([0, 0, z_0])
    loss_event = events.OutOfRangeSphere(limit, center=center)

    # Simulation
    POINTS = 10
    t_0 = 0
    t_end = 200E-3
    dt = 1E-6
    PARTICLES = 1E5
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

    sim.pickle(path='output.pickle')


if __name__ == '__main__':
    main()
