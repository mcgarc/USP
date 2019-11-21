import unittest
import numpy as np
import scipy.constants as spc

from wire import WireSegment

class TestWireMethods(unittest.TestCase):

    def setUp(self):
        """
        Initialise a wire for basic testing
        """
        self.start = [-1, 0, 0]
        self.end = [1, 0, 0]
        self.current = 1
        self.wire = WireSegment(self.start, self.end, self.current)

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

    def test_field(self):
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
        


if __name__ == '__main__':
    unittest.main()

