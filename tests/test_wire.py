import unittest
import numpy as np
import scipy.constants as spc

import wire
import parameter

class TestWireSegmentMethods(unittest.TestCase):
    """
    Test wire segment methods with a constant current
    """

    def setUp(self):
        """
        Initialise a wire for basic testing
        """
        self.start = [-1, 0, 0]
        self.end = [1, 0, 0]
        self.current = parameter.ConstantParameter(1)
        self.wire = wire.WireSegment(self.current, self.start, self.end)

    def test_properties(self):
        """
        Basic testing of the wire property functions and that initialisation ran
        correctly
        """
        np.testing.assert_array_equal(self.wire.start, np.array(self.start))
        np.testing.assert_array_equal(self.wire.end, np.array(self.end))
        self.assertEqual(self.wire.current, self.current)

    def test_equality(self):
        same_wire = wire.WireSegment(self.current, self.start, self.end)
        self.assertEqual(self.wire, same_wire)
        # Check with same current
        same_current = parameter.ConstantParameter(self.current.value(0))
        same_wire = wire.WireSegment(same_current, self.start, self.end)
        self.assertEqual(self.wire, same_wire)
        # Check with different current
        diff_current = parameter.ConstantParameter(-2)
        different_wire = wire.WireSegment(diff_current, [-5, 0, 0], [0, 0, 3])
        self.assertNotEqual(self.wire, different_wire)

    def test_set_current(self):
        """
        Cannot set int as current (must be of type AbstractParameterProfile)
        """
        with self.assertRaises(ValueError):
            self.wire.set_current(1)
        cur = 5
        new_current = parameter.ConstantParameter(cur)
        self.wire.set_current(new_current)
        self.assertEqual(self.wire.current.value(0), cur)

    def test_field_simple(self):
        """
        Test the field method.

        This is a simple test case, direction should be in z, and the magnitude
        can be found analytically
        """
        r = [0, 1, 0]
        magnitude = spc.mu_0 / (2 * np.sqrt(2) * np.pi)
        z_hat = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.wire.field(0, r),
                                             magnitude * z_hat
                                             )

    def test_field_simple_reverse(self):
        """
        Test the field method when the wire current is reversed
        """
        current_nve = parameter.ConstantParameter(-1)
        self.wire.set_current(current_nve)
        r = [0, 1, 0]
        magnitude = spc.mu_0 / (2 * np.sqrt(2) * np.pi)
        z_hat = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.wire.field(0, r),
                                             -1 * magnitude * z_hat
                                             )


class TestWireClusterMethods(unittest.TestCase):

    def setUp(self):
        # setup parallel wire cluster
        current_pve = parameter.ConstantParameter(1)
        current_nve = parameter.ConstantParameter(-1)
        wire_top = wire.WireSegment(current_pve, [-1, 1, 0], [1, 1, 0])
        wire_bot = wire.WireSegment(current_pve, [-1,-1, 0], [1,-1, 0])
        self.cluster_parallel = wire.WireCluster([wire_top, wire_bot])
        # And anti-parallel
        wire_top = wire.WireSegment(current_pve, [-1, 1, 0], [1, 1, 0])
        wire_bot = wire.WireSegment(current_nve, [-1,-1, 0], [1,-1, 0])
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
        np.testing.assert_array_almost_equal(self.cluster_parallel.field(0, r),
                                             magnitude * z_hat
                                             )

    def test_field_antiparalell_simple(self):
        """
        Field at the origin for antiparallel case is zero
        """
        r = [0, 0, 0] # origin
        field_origin = self.cluster_antiparallel.field(0, r)
        np.testing.assert_array_almost_equal(field_origin, np.zeros(3))

class TestZWireMethods(unittest.TestCase):

    def setUp(self):
        self.end_length = 10
        self.current = parameter.ConstantParameter(1)
        self.al = 1.
        self.z_wire = wire.ZWire(self.current,
                                 self.al,
                                 end_length=self.end_length
                                 )

    def test_init(self):
        """
        Test that the zwire initialises in the expected way
        """
        # TODO Check non-axial wires too
        axis = self.z_wire.wires[1]
        axis_expect = wire.WireSegment(self.current,
                                       [0, -self.al/2, 0],
                                       [0, self.al/2, 0]
                                       )
        self.assertEqual(axis, axis_expect)

    def test_field_direction(self):
        """
        Check that the field above the origin has no z component
        """
        heights = np.logspace(-5, 1, 10)
        for h in heights:
            z_field = self.z_wire.field(0, [0, 0, h])[2]
            self.assertEqual(z_field, 0)
