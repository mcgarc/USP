import psa
from USP.consts import u_B

def main():
    z_axis = 10E-3
    I = 20*u_B
    z_0 = 1E-3
    trap = psa.z_trap(I_0=I, z_axis=z_axis, z_0=z_0)
    psa.psa(
        trap,
        T=.6E-3,
        r_init=[0, 0, z_0],
        x_range=(-8E-3, 8E-3),
        y_range=(-2E-3, 2E-3,
        z_range=(0, 4E-3),
        loss_limit=8E-3,
        t_end = 300E-3,
        dt = 1E-5
        )

if __name__ == '__main__':
    main()
