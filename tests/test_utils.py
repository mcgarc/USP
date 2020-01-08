import unittest
import numpy as np

import utils

class TestUtils(unittest.TestCase):
    """
    Tests for basic utilities
    """

    def test_clean_vector(self):
        """
        """
        vect = [1, 2, 3]
        clean = utils.clean_vector(vect)
        self.assertEqual(type(clean), np.ndarray)
        np.testing.assert_array_equal(np.array(vect), clean)
        self.assertRaises(ValueError, utils.clean_vector, vect, 4)

        # Not 3-vectors
        vect_2 = [1, 2]
        clean_2 = utils.clean_vector(vect_2, 2)
        np.testing.assert_array_equal(vect_2, clean_2)
