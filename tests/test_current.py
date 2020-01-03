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

class TestRampCurrent(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        For most tests we can just use a simple ramp that has values
        [linear ramp 0 to 1] for 0<=t<=1,
        1 for 1<=t<=2,
        [linear ramp 1 to 0] for 2<=t<=3
        and 0 otherwise
        """
        self.curs = [1]
        self.times = [0, 1, 2, 3]
        self.ramp = current.RampCurrent(self.curs, self.times)

    def test_init(self):
        self.assertEqual(self.curs, self.ramp.currents)
        self.assertEqual(self.times, self.ramp.times)
        self.assertEqual(3, len(self.step))

    def test_eq(self):
        self.assertTrue(self.ramp == self.ramp)
        diff_ramp = current.RampCurrent([7], [10, 11, 12])
        self.assertFalse(self.ramp == diff_ramp)

    def test_current_simple(self):
        self.assertEqual(self.ramp.current(-1), 0)
        self.assertEqual(self.ramp.current(0), 0)
        self.assertEqual(self.ramp.current(0.5), 0.5)
        self.assertEqual(self.ramp.current(1), 1)
        self.assertEqual(self.ramp.current(1.2), 1)
        self.assertEqual(self.ramp.current(2), 1)
        self.assertEqual(self.ramp.current(2.5), 0.5)
        self.assertEqual(self.ramp.current(2.9), 0.1)
        self.assertEqual(self.ramp.current(3), 0)
        self.assertEqual(self.ramp.current(3.5), 0)

    def test_current_two_stage(self):
        """
        Test for a two stage ramp current that takes values
        [linear ramp 0 to 1] for 0<=t<=1,
        1 for 1<=t<=2,
        [linear ramp 1 to 2] for 1<=t<=2,
        2 for 2<=t<=3,
        [linear ramp 2 to 0] for 3<=t<=4
        and 0 otherwise
        """
        curs = [1, 2]
        times = [0, 1, 2, 3, 4, 5]
        ramp = current.RampCurrent(curs, times)
        self.assertEqual(ramp.current(-1), 0)
        self.assertEqual(ramp.current(0), 0)
        self.assertEqual(ramp.current(0.5), 0.5)
        self.assertEqual(ramp.current(1), 1)
        self.assertEqual(ramp.current(1.2), 1)
        self.assertEqual(ramp.current(2), 1)
        self.assertEqual(ramp.current(2.5), 1.5)
        self.assertEqual(ramp.current(2.9), 1.9)
        self.assertEqual(ramp.current(3), 2)
        self.assertEqual(ramp.current(3.8), 2)
        self.assertEqual(ramp.current(4), 2)
        self.assertEqual(ramp.current(4.1), 1.8)
        self.assertEqual(ramp.current(4.5), 1)
        self.assertEqual(ramp.current(5), 0)
        self.assertEqual(ramp.current(6), 0)
