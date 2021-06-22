"""
Tranfer sim as used in LSR
"""
from USP import *
import numpy as np

displacements = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
#times = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
times = [2.5, 7.5, 12.5, 45, 60, 75, 100, 125]

for t in times:
    print(t)


    # Times
    t_m3 = 0E-3 # Init time (t_-3)
    t_m2 = 50E-3 # Internal trap compression ramp start time
    t_m1 = 150E-3  #  compresion ramp end time
    t_0 = 200E-3 # start transfer
    t_ramp = t * 1E-3
    t_1 = t_0 + t_ramp # end transfer
    t_hold_ext = 150E-3 - t_ramp
    t_end = t_1 + t_hold_ext
    
    # params
    mass = consts.m_CaF
    
    # Initialise internal QP
    b_0_0 = consts.u_B * .1 # Levitation graident 10G/cm
    b_0_1 = consts.u_B * .3 # Internal trap gradient 30G/cm
    b_0 = parameter.RampParameterUpDown2(-2, -1, t_m2, t_m1, t_0, t_1, b_0_0,  b_0_1)
    #b_0 = parameter.ConstantParameter(b_0_1)
    internal_field = field.QuadrupoleField(b_0)
    internal_trap = trap.FieldTrap(internal_field.field)
    
    # Initialise extermal QP (Magnetic Transport Trap - MTT)
    b_1_0 = consts.u_B * .6 # MTT gradient 60G/cm
    b_1 = parameter.RampParameterUpDown(t_0, t_1, 1, 2, b_1_0)
    external_field = field.QuadrupoleField(b_1, r_0=[3E-3, 0, 0])
    external_trap = trap.FieldTrap(external_field.field)
    
    combi_trap = trap.SuperimposeTrap([internal_trap, external_trap])
    #combi_trap = trap.SuperimposeTrap([internal_trap])
    
    # Particle generators
    l = 1E-3
    r_gen = rand.NormalGenerator([0, 0, 0], [l, l, l])
    T = 5E-6
    v_gen = rand.TemperatureGeneratorNormal(mass, T, length=3)
    
    # Simulation
    POINTS = 200
    dt = 1E-4
    PARTICLES = 100
    sim = simulation.Simulation(
            combi_trap,
            t_m3,
            t_end,
            dt,
            POINTS,
            process_no=4
            )
    sim.init_particles(PARTICLES, mass, r_gen, v_gen)
    
    sim.run()
    
    #sim.pickle(path=f'results/ramp_int_mtt.pickle')
    sim.pickle(path=f'results/disp/ramp_int_mtt_time{t}.pickle')
