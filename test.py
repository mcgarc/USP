import unittest
import numpy as np
import scipy.constants as spc

import wire

class TestWireMethods(unittest.TestCase):

    def setUp(self):
        """
        Initialise a wire for basic testing
        """
        self.start = [-1, 0, 0]
        self.end = [1, 0, 0]
        self.current = 1
        self.wire = wire.WireSegment(self.start, self.end, self.current)

    def tearDown(self):
        pass

    def test_wire_properties(self):
        """
        Basic testing of the wire property functions and that initialisation ran
        correctly
        """
        np.testing.assert_array_equal(self.wire.start, np.array(self.start))
        np.testing.assert_array_equal(self.wire.end, np.array(self.end))
        self.assertEqual(self.wire.current, self.current)

    def test_equality(self):
        same_wire = wire.WireSegment(self.start, self.end, self.current)
        self.assertEqual(self.wire, same_wire)
        different_wire = wire.WireSegment([-5, 0, 0], [0, 0, 3], -2)
        self.assertNotEqual(self.wire, different_wire)

    def test_field_simple(self):
        """
        Test the field method.

        This is a simple test case, direction should be in z, and the magnitude
        can be found analytically
        """
        r = [0, 1, 0]
        magnitude = spc.mu_0 / (2 * np.sqrt(2) * np.pi)
        z_hat = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.wire.field(r),
                                             magnitude * z_hat
                                             )

    def test_field_simple_reverse(self):
        """
        Test the field method when the wire current is reversed
        """
        self.wire.set_current(-1)
        r = [0, 1, 0]
        magnitude = spc.mu_0 / (2 * np.sqrt(2) * np.pi)
        z_hat = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.wire.field(r),
                                             -1 * magnitude * z_hat
                                             )


class TestWireClusterMethods(unittest.TestCase):

    def setUp(self):
        # setup parallel wire cluster
        wire_top = wire.WireSegment([-1, 1, 0], [1, 1, 0], 1)
        wire_bot = wire.WireSegment([-1,-1, 0], [1,-1, 0], 1)
        self.cluster_parallel = wire.WireCluster([wire_top, wire_bot])
        # And anti-parallel
        wire_top = wire.WireSegment([-1, 1, 0], [1, 1, 0], 1)
        wire_bot = wire.WireSegment([-1,-1, 0], [1,-1, 0], -1)
        self.cluster_antiparallel = wire.WireCluster([wire_bot, wire_top])

    def test_equality(self):
        self.assertEqual(self.cluster_parallel, self.cluster_parallel)
        self.assertEqual(self.cluster_antiparallel, self.cluster_antiparallel)
        self.assertNotEqual(self.cluster_parallel, self.cluster_antiparallel)

    def test_field_parallel_simple(self):
        """
        Field at origin for parallel case can be determined analytically
        """
        r = [0, 0, 0] # origin
        magnitude = spc.mu_0 / (np.sqrt(2) * np.pi) # analytically deterimed
        z_hat = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.cluster_parallel.field(r),
                                             magnitude * z_hat
                                             )

    def test_field_antiparalell_simple(self):
        """
        Field at the origin for antiparallel case is zero
        """
        r = [0, 0, 0] # origin
        field_origin = self.cluster_antiparallel.field(r)
        np.testing.assert_array_almost_equal(field_origin, np.zeros(3))


if __name__ == '__main__':
    unittest.main()

