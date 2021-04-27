from USP import *
import numpy as np
import time



def main():
    """
    Test simulation of a few particles in a Zeeman guide
    """

    # Initialise Zeeman guide
    z_guide = trap.ZeemanGuide()

    # Particle generators
    l = 1E-3
    rsigma = 1E-3
    T = 50E-6
    vsigma = np.sqrt(2 * consts.k_B * np.abs(T) / consts.m_CaF)
    vz = 144
    r_gen = rand.NormalGenerator([0, 0, 3E-3], [rsigma, rsigma, rsigma])
    v_gen = rand.NormalGenerator([0, 0, vz], [vsigma, vsigma, vsigma])

    # Simulation
    t_end = 10E-3
    dt = 1E-5
    samples = 100
    mass = consts.m_CaF
    T = 4
    v_spread = np.sqrt(2*consts.k_B*T/mass)
    v_z = 144
    sim = simulation.Simulation(z_guide, 0, t_end, dt, samples)
    sim.init_particles(30, mass, r_gen, v_gen)
    sim.run()

    # Diagnosis
    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

if __name__ == '__main__':
    main()
