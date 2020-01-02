import unittest
import numpy as np
import scipy.constants as spc

import trap
import wire
import current


class TestTrap(unittest.TestCase):
    """
    Test trap
    """

    def test_zwire_rt_static(self):
        """
        Test the static zwire trap generated with position and time. Check that
        the zero is where we expect it to be
        """
        cur = current.ConstantCurrent(1)
        zwire = wire.ZWire(cur, 0.1) 
        zero_pos = [0, 0, 0.1]
        ztrap = trap.ClusterTrapStaticTR(zwire, 0, zero_pos)
        # Test zero at time 0
        zero_value = ztrap.field(0, zero_pos)
        np.testing.assert_array_almost_equal(zero_value, np.zeros(3))
        # Test zero at later times (there should be no time evolution
        zero_value = ztrap.field(1, zero_pos)
        np.testing.assert_array_almost_equal(zero_value, np.zeros(3))
        # Test that we have a non-zero value elsewhere
        nonzero_value = ztrap.field(0, [0.1, 0.1, 0.1])
        self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                nonzero_value,
                np.zeros(3)
                )

