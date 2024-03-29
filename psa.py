"""
Phase space acceptance simulation as used in LSR
"""
from USP import *
import time
import numpy as np
import os

z_00 = 3E-3
y_00 = 2E-3
I_00 = 100 * consts.u_B
u_axis0 = 16E-3
u_center0 = [0, 0, -4E-3]

def u_trap(I_0=I_00, z_0=z_00, u_axis=u_axis0, center=u_center0):
    """
    U trap as we hold after coil transfer
    """
    I = parameter.ConstantParameter(I_0)
    z = parameter.ConstantParameter(z_0)
    u_wire = wire.UWire(I, u_axis, center=center)
    u_trap = trap.ClusterTrap(u_wire, z)
    return u_trap

I_01 = 20 * consts.u_B
z_01 = 1E-3
z_axis0 = 2E-3

def z_trap(I_0=I_01, z_0=z_01, z_axis=z_axis0):
    """
    Z trap as we hold after U transfer
    """
    I = parameter.ConstantParameter(I_0)
    z = parameter.ConstantParameter(z_0)
    z_wire = wire.ZWireXAxis(I, z_axis)
    z_trap = trap.ClusterTrap(z_wire, [0, 0, z], bias_scale=[0.95, 1, 0])
    return z_trap
    
x_range = (-10E-3, 10E-3)
y_range = (-8E-3, 8E-3)
z_range = (0, 6E-3)
v_range = (-1, 1)
r_init = [0, 0, z_00]
PARTICLES = 1E5
POINTS = 30

def psa(
        trap,
        N_particles=PARTICLES,
        r_init=r_init,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        vx_range=v_range,
        vy_range=v_range,
        vz_range=v_range,
        loss_limit=8E-3,
        t_end = 300E-3,
        dt = 1E-5,
        process_no=32,
        sample_points = POINTS,
        output_name='output'
        ):
    """
    Test simulation of particles transferring between traps
    """
    
    # Initial conditions
    mass = consts.m_CaF
    
    # Particle generators
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    vx0, vx1 = vx_range
    vy0, vy1 = vy_range
    vz0, vz1 = vz_range
    r_gen = rand.UniformGenerator([x0, y0, z0], [x1, y1, z1])
    v_gen = rand.UniformGenerator([vx0, vy0, vz0], [vx1, vy1, vz1])

    # Particle loss
    center = parameter.ArrayParameter(r_init)
    loss_event = events.OutOfRangeSphere(loss_limit, center=center)

    # Simulation
    t_0 = 0
    sim = simulation.Simulation(
            trap,
            t_0,
            t_end,
            dt,
            POINTS,
            events=loss_event,
            process_no=process_no
            )
    sim.init_particles(N_particles, mass, r_gen, v_gen)

    sim.run()

    sim.pickle(path=f'{output_name}.pickle')


if __name__ == '__main__':
    pass
