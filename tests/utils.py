"""
Utilities for testing
"""

import utils

def harmonic_potential(t, r, k, r_0=[0, 0, 0]):
    """
    A harmonic potential. Params: t time (time indep. potential but important
    for compatibility), r position, k spring constant, r_0 potential centre
    """
    r = utils.clean_vector(r)
    r_0 = utils.clean_vector(r_0)
    return 0.5*k*np.dot(r - r_0, r - r_0)


