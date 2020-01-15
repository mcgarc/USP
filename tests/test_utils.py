import unittest
import numpy as np

from USP import utils

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

    def test_grad_direction(self):
        """
        Test the grad method gives correct values for simple test function and
        raises value error for bad direction.
        """
        
        # Test gradient function
        def f(t, r):
            return r[0] + 2 * r[1] + 3 * r[2]

        t_0 = 0
        r_0 = [0, 0, 0]
        # Return correct method for valid directions
        # x
        self.assertEqual(1, utils.grad(f, t_0, r_0, 'x'))
        self.assertEqual(1, utils.grad(f, t_0, r_0, 0))
        # y
        self.assertEqual(2, utils.grad(f, t_0, r_0, 'y'))
        self.assertEqual(2, utils.grad(f, t_0, r_0, 1))
        # z
        self.assertEqual(3, utils.grad(f, t_0, r_0, 'z'))
        self.assertEqual(3, utils.grad(f, t_0, r_0, 2))
        # Raise value errors for invalid directions
        test_direction_fail = [4, '1', 'q']
        for direction in test_direction_fail:
            self.assertRaises(ValueError, utils.grad, f, t_0, r_0, direction) 
