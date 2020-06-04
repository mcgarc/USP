import psa

def main():
    trap = psa.u_trap()
    psa.psa(trap, N_particles=100, process_no=8)

if __name__ == '__main__':
    main()
