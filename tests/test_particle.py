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
        self.particle = particle.Particle(self.r, self.v, self.m)

    def setup_integ_test_trivial(self):
        """
        Basic quadrupole field trapping a single particle
        """
        qp = field.QuadrupoleField(1)
        qp_trap = trap.FieldTrap(qp.field)
        t_end = 1
        max_step = t_end/1000
        self.particle.integ(qp_trap.potential, t_end, max_step=max_step)

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
        self.assertIsNone(self.particle._integ)
        self.setup_integ_test_trivial()
        self.assertIsNotNone(self.particle._integ)

    def test_integ_trivial(self):
        """
        """
        self.setup_integ_test_trivial()

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
