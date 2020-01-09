import unittest
import numpy as np

import particle

class TestParticle(unittest.TestCase):
    """
    Tests the particle class, including ODE solver
    """

    def setUp(self):
        self.r = [0, 0, 0]
        self.v = [0, 0, 0]
        self.m = 1
        self.particle = particle.Particle(self.r, self.v, self.m)


    def test_properties(self):
        np.testing.assert_array_equal(self.r, self.particle.r)
        np.testing.assert_array_equal(self.v, self.particle.v)
        self.assertEqual(self.m, self.particle.m)
        self.assertIsNone(self.particle.t) # No integ defined yet
        Q = np.concatenate((self.r, self.v))
        np.testing.assert_array_equal(Q, self.particle.Q)

    def test__dQ_dt(self):
        """
        TODO: Test _dQ_dt, for finding the time derivative of Q in a specified
        potential
        """
        pass

    def test_init_integ(self):
        """
        TODO: Test initialisation of the integrator for a simple potential
        """
        pass

    def test_integ_properties(self):
        """
        TODO: Test that once _integ is not None, its properties can be
        reclaimed
        """
        #self.assertIsNotNone(self.particle.t) # Maybe this is 0?
        pass

    def test_step_integ(self):
        """
        TODO: Test stepping of a particle's integrator
        """
        pass

    def test_check_termination(self):
        """
        Test that a terminated particle cannot be stepped further
        """
        self.assertFalse(self.particle.terminated)
        self.particle.terminate()
        self.assertTrue(self.particle.terminated)
        # If terminated then particle._integ will not be accessed, so there will
        # not be an error
        try:
            self.particle.step_integ()
        except AttributeError:
            self.fail('Attempted propagation of terminated particle')


    def test_check_termination_conditions(self):
        """
        TODO: Write specification for termination conditions and write tests for
        development
        """
        pass
