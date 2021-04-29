import unittest
import numpy as np

from USP import parameter
from USP import trap
from USP import field
from USP import particle

class TestParticle(unittest.TestCase):
    """
    Tests the particle class, including ODE solver
    """

    def setUp(self):
        self.r = [0, 0, 0]
        self.v = [0, 0, 0]
        self.m = 1
        self.t_0 = 0
        self.t_end = 1
        self.dt = 1E-3
        self.points = 1000
        self.particle = particle.Particle(
                self.r,
                self.v,
                self.m,
                self.t_0,
                self.t_end,
                self.dt,
                self.points
                )

    def test_properties(self):
        self.assertEqual(self.m, self.particle.m)

    def test_position_velocity(self):
        np.testing.assert_array_equal(self.r, self.particle.r(0))
        np.testing.assert_array_equal(self.v, self.particle.v(0))
        Q = np.concatenate((self.r, self.v))
        np.testing.assert_array_equal(Q, self.particle.Q(0))

    def test_postition_velocity_before_integ(self):
        """
        Calling r, v and Q for non-zero times when no integration has been
        perfromed should return an error
        """
        pass
        

    def test_postition_velocity_after_integ(self):
        """
        Calling r, v and Q for non-zero times when no integration has been
        perfromed should return 3-vectors of position
        """
        pass

    def test__dQ_dt(self):
        """
        TODO: Test _dQ_dt, for finding the time derivative of Q in a specified
        potential
        """
        pass

    def test_init_integ(self):
        """
        Test initialisation of the integrator for a simple potential
        """
        #

    def test_integ_trivial(self):
        """
        """
        pass

    def test_check_termination(self):
        """
        Test that a terminated particle is flagged as such
        """
        pass

    def test_check_termination_conditions(self):
        """
        TODO: Write specification for termination conditions and write tests for
        development
        """
        pass
