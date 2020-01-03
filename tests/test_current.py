import unittest
import numpy as np
import scipy.constants as spc

import current


class TestStepCurrent(unittest.TestCase):
    """
    Test step current
    """

    def setUp(self):
        """
        For most tests we can just use a simple step that has values
        1 for 0<=t<1, 2 for 1<=t<2, 3 for 2<=t<3, and 0 otherwise
        """
        self.curs = [1, 2, 3]
        self.times = [0, 1, 2, 3]
        self.step = current.StepCurrent(self.curs, self.times)

    def test_init(self):
        self.assertEqual(self.curs, self.step.currents)
        self.assertEqual(self.times, self.step.times)
        self.assertEqual(3, len(self.step))

    def test_eq(self):
        self.assertTrue(self.step == self.step)
        diff_step = current.StepCurrent([7,8,9], [10, 11, 12, 13])
        self.assertFalse(self.step == diff_step)

    def test_current(self):
        """
        Test that the current method of StepCurrent returns the expected value
        for various times
        """
        self.assertEqual(self.step.current(-1), 0)
        self.assertEqual(self.step.current(0), 1)
        self.assertEqual(self.step.current(0.1), 1)
        self.assertEqual(self.step.current(1), 2)
        self.assertEqual(self.step.current(1.2), 2)
        self.assertEqual(self.step.current(2), 3)
        self.assertEqual(self.step.current(2.9), 3)
        self.assertEqual(self.step.current(3), 0)
        self.assertEqual(self.step.current(3.1), 0)
