import unittest
import numpy as np

import parameter
import trap
import wire
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

    def trivial_ztrap(self):
        """
        Return a traivial z-trap with zero potential
        """
        cur = parameter.ConstantParameter(0)
        height = parameter.RampParameter([0.1, 0.2], [-2, -1, 50, 1050, float('inf'), float('inf')])
        zcluster = wire.ZWire(cur, 0.1)
        ztrap = trap.ClusterTrap(
                zcluster,
                height
                )
        return ztrap

    def setup_integ_test_trivial(self):
        """
        Create a trivial potential from a zwire for use in testing the particle
        ODE solver
        """
        ztrap = self.trivial_ztrap()
        # Test with a particle that won't be inside the wire
        self.particle.set_r([0, 0, 0.1]) 
        # Integrator parameters  and init
        t_end = 1
        max_step = t_end/1000
        self.particle.init_integ(ztrap.potential, t_end, max_step=max_step)

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
        Test initialisation of the integrator for a simple potential
        """
        self.assertIsNone(self.particle._integ)
        self.setup_integ_test_trivial()
        self.assertIsNotNone(self.particle._integ)
        self.assertEqual(0, self.particle.t)

    def test_integ_trivial(self):
        """
        """
        self.setup_integ_test_trivial()

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
