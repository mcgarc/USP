"""
Tranfer sim as used in LSR
"""
from USP import *
import numpy as np


# Times
t_m3 = -200E-3 # Init time (t_-3)
t_m2 = -150E-3 # MTT compression ramp start time
t_m1 = -50E-3  # MTT compresion ramp end time
t_0 = 0
t_settle = 100E-3 # Time for molecules to settle in QP
t_1 = t_0 + t_settle
t_ramp = 100E-3 # Duration of ramp QP to U
t_2 = t_1 + t_ramp
t_settle = 100E-3 # Time for molecules to settle in U
t_3 = t_2 + t_settle
t_ramp = 100E-3 # Duration of ramp U to Z0i
t_4 = t_3 + t_ramp
t_settle = 100E-3 # Settle time in Z0i
t_5 = t_4 + t_settle
t_ramp = 100E-3 # Duration of compression ramp
t_6 = t_5 + t_ramp
t_settle = 100E-3 # Settle time in Z0f
t_7 = t_6 + t_settle

# params
mass = consts.m_CaF

# Initialise QP
b_1_0 = consts.u_B * .1 # Levitation graident 10G/cm
b_1_1 = consts.u_B * .6
b_1 = parameter.RampParameter([b_1_0, b_1_1], [-2, -1, t_m2, t_m1, t_1, t_2])
qp = field.QuadrupoleField(b_1, r_0=[0, 0, 3E-3])
qp_trap = trap.FieldTrap(qp.field)

# Initialise U wire
I_1 = -100 * consts.u_B
I = parameter.RampParameter([I_1], [t_1, t_2, t_3, t_4])
u_axis = 16E-3
u_wire = wire.UWire(I, u_axis, center=[0,0,-4E-3])
#B_y = parameter.RampParameter([-22E-4*consts.u_B], [t_1, t_2, 1, 2])
#B_z = parameter.RampParameter([-14E-4*consts.u_B], [t_1, t_2, 1, 2])
#bias = parameter.ArrayParameter([0, B_y, B_z])
u_trap = trap.ClusterTrapStatic(u_wire)

# Initialise Z wire
I_z_0 = -30 * consts.u_B
I_z = parameter.RampParameter([I_z_0], [t_3, t_4, 1, 2])
z_axis = 12E-3
z_wire = wire.ZWireXAxis(I_z, z_axis)
z_trap = trap.ClusterTrapStatic(z_wire)

# Combination trap
height = parameter.RampParameter([3E-3, 1E-3], [-2, -1, t_5, t_6, 2, 3])
center = [0, 0, height]
center = parameter.ArrayParameter(center)
combi_trap = trap.SuperimposeTrapWBias([qp_trap, u_trap, z_trap], center, bias_scale=[0.95, 1, 1])

# Particle generators
l = 1E-3
r_gen = rand.NormalGenerator([0, 0, 3E-3], [l, l, l])
#v_gen = rand.TemperatureGenerator(mass, 100E-6, length=3)
T = 50E-6
v_gen = rand.TemperatureGeneratorNormal(mass, T, length=3)

# Particle loss (if particle touches the chip or reaches a limit in a cardinal direction)
limit = 10E-3
loss_event = events.OutOfRangeBox(-limit, -limit, -limit, limit, limit, limit)

# Simulation
POINTS = 200
t_end = t_7
dt = 1E-4
PARTICLES = 1E3
sim = simulation.Simulation(
        combi_trap,
        t_m3,
        t_end,
        dt,
        POINTS,
        events=loss_event,
        process_no=32
        )
sim.init_particles(PARTICLES, mass, r_gen, v_gen)

sim.run()

sim.pickle(path='output.pickle')
