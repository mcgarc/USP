from USP import *
import numpy as np

# Times
t_m3 = -200E-3 # Init time (t_-3)
t_m2 = -150E-3 # MTT compression ramp start time
t_m1 = -50E-3  # MTT compresion ramp end time
t_0 = 0
t_settle = 200E-3 # Time for molecules to settle in QP
t_1 = t_0 + t_settle
t_ramp = 200E-3 # Duration of ramp QP to U
t_2 = t_1 + t_ramp
t_settle = 400E-3 # Timr for molecules to settle in U
t_3 = t_2 + t_settle

# params
mass = consts.m_CaF

# Initialise QP
b_1_0 = consts.u_B * .1 # Levitation graident 10G/cm
b_1_1 = consts.u_B * .6
b_1 = parameter.RampParameter([b_1_0, b_1_1], [-2, -1, t_m2, t_m1, t_1, t_2])
qp = field.QuadrupoleField(b_1, r_0=[0, 0, 3E-3])
qp_trap = trap.FieldTrap(qp.field)

# Initialise Z wire with ramp up after init
I_1 = -100 * consts.u_B
I = parameter.RampParameter([I_1], [t_1, t_2, 1, 2])
u_axis = 16E-3
u_wire = wire.UWire(I, u_axis, center=[0,0,-4E-3])
u_trap = trap.ClusterTrapStatic(u_wire)

# Combination trap
center = [0, 0, 3E-3]
center = parameter.ArrayParameter(center)
combi_trap = trap.SuperimposeTrapWBias([qp_trap, u_trap], center, bias_scale=[0,1,1])

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
t_end = t_3
dt = 1E-3
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
