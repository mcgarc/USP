from USP import *
import numpy as np
import time

def harmonic_2d_potential(t, r):
    V_0 = 0
    m = 1
    omega_0 = 1
    return V_0 + 0.5 * m * omega_0 * (r[1]*r[1] + r[2]*r[2])

def axial_B_sin(z, L):
    z = float(z)
    L = float(L)
    return np.sin(np.pi*z/L)**2

def radial_strong_B(rho, B0, B1, R):
    rho = float(rho)
    B0 = float(B0)
    B1 = float(B1)
    R = float(R)
    return (B1 - B0)*rho**2/R**2

def radial_weak_B(rho, B0, B1, B2, R):
    rho = float(rho)
    B0 = float(B0)
    B1 = float(B1)
    B2 = float(B2)
    R = float(R)
    # a soln
    det_a = B0*B1 - B0*B2 - B1*B2 + B2**2
    # ignore very small values (floating point happens)
    if det_a < 1E-10:
        det_a = 0
    a = ((B0 + B1 - 2*B2)*R**4 + 2*np.sqrt(det_a)*R**4)/R**8
    # b soln
    det_b = (B1-B2)*(B0-B2)
    if det_b < 0:
        print(B0, B1, B2, det_b)
    b = 2*(B2 - B0 - np.sqrt(det_b))/R**2
    return a*rho**4 + b*rho**2


def zeeman_potential(t, r):
    x = r[0]
    y = r[1]
    z = r[2]
    B0str = 1.05
    B1strX = 1.25
    B0strY = 1.
    B0weak = 0.05
    B1weakX = 0.3
    B1weakY = 0.5
    B2weakX = 0.04
    B2weakY = 0.05
    L = 2E-2
    R = 2.5E-3
    pot = axial_B_sin(z, L) * (
            radial_strong_B(x, B0str, B1strX, R) +
            radial_strong_B(y, B0str, B0strY, R) +
            B0str
            ) + axial_B_sin(z + L/2, L) * (
            radial_weak_B(x, B0weak, B1weakX, B2weakX, R) +
            radial_weak_B(y, B0weak, B1weakY, B2weakY, R) +
            B0weak
            )
    return pot * consts.u_B


def main():
    """
    Test simulation of a few particles in a rising potential
    """

    # Initialise harmonic trap
    h_trap = trap.AbstractPotentialTrap(zeeman_potential)

    # Simulation
    t_end = 10E-3
    mass = consts.m_CaF
    hole_diameter = 4E-3
    r_spread = hole_diameter/4
    T = 4
    v_spread = np.sqrt(2*consts.k_B*T/mass)
    v_z = 144
    sim = simulation.Simulation(4, h_trap, 0, t_end, 1E-6)
    sim.init_particles(100, mass, r_spread, v_spread, v_centre=[0, 0, v_z])
    sim.run()
    print(f'runtime: {sim.run_time}')

    print(sim.get_total_energy(0))
    print(sim.get_total_energy(t_end))

    # Visualisation
    sim.plot_start_end_positions()

#    sim.plot_phase_diagram(0, 0, 100, time_gradient=True, colorbar=True,
#            output_path='results/phasespace_x')

    sim.animate(500, output_path='results/zeeman.mp4')

if __name__ == '__main__':
    main()
