import unittest
import numpy as np
import scipy.constants as spc

import current


class TestStepCurrent(unittest.TestCase):
    """
    Test step current
    """

    def setUp(self):
        self.curs = [1, 2, 3]
        self.times = [0, 1, 2, 3]
        self.step = current.StepCurrent(self.curs, self.times)

    def test_init(self):
        pass

    def test_make_current_differences(self):
        pass

    def test_eq(self):
        pass

    def test_current(self):
        """
        Test the current method of StepCurrent
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
