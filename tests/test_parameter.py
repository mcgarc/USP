import unittest
import numpy as np
import scipy.constants as spc

from USP import parameter


class TestStepParameter(unittest.TestCase):
    """
    Test step parameter
    """

    def setUp(self):
        """
        For most tests we can just use a simple step that has values
        1 for 0<=t<1, 2 for 1<=t<2, 3 for 2<=t<3, and 0 otherwise
        """
        self.vals = [1, 2, 3]
        self.times = [0, 1, 2, 3]
        self.step = parameter.StepParameter(self.vals, self.times)

    def test_init(self):
        self.assertEqual(self.vals, self.step.values)
        self.assertEqual(self.times, self.step.times)
        self.assertEqual(3, len(self.step))

    def test_eq(self):
        self.assertTrue(self.step == self.step)
        diff_step = parameter.StepParameter([7,8,9], [10, 11, 12, 13])
        self.assertFalse(self.step == diff_step)

    def test_value(self):
        """
        Test that the value method of StepParameter returns the expected value
        for various times
        """
        self.assertEqual(self.step.value(-1), 0)
        self.assertEqual(self.step.value(0), 1)
        self.assertEqual(self.step.value(0.1), 1)
        self.assertEqual(self.step.value(1), 2)
        self.assertEqual(self.step.value(1.2), 2)
        self.assertEqual(self.step.value(2), 3)
        self.assertEqual(self.step.value(2.9), 3)
        self.assertEqual(self.step.value(3), 0)
        self.assertEqual(self.step.value(3.1), 0)

class TestRampParameter(unittest.TestCase):
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
        self.vals = [1]
        self.times = [0, 1, 2, 3]
        self.ramp = parameter.RampParameter(self.vals, self.times)

    def test_init(self):
        self.assertEqual(self.vals, self.ramp.values)
        self.assertEqual(self.times, self.ramp.times)
        self.assertEqual(1, len(self.ramp))

    def test_eq(self):
        self.assertTrue(self.ramp == self.ramp)
        diff_ramp = parameter.RampParameter([7], [10, 11, 12, 13])
        self.assertFalse(self.ramp == diff_ramp)

    def test_parameter_simple(self):
        self.assertEqual(self.ramp.value(-1), 0)
        self.assertEqual(self.ramp.value(0), 0)
        self.assertEqual(self.ramp.value(0.5), 0.5)
        self.assertEqual(self.ramp.value(1), 1)
        self.assertEqual(self.ramp.value(1.2), 1)
        self.assertEqual(self.ramp.value(2), 1)
        self.assertEqual(self.ramp.value(2.5), 0.5)
        self.assertAlmostEqual(self.ramp.value(2.9), 0.1)
        self.assertEqual(self.ramp.value(3), 0)
        self.assertEqual(self.ramp.value(3.5), 0)

    def test_parameter_two_stage(self):
        """
        Test for a two stage ramp parameter that takes values
        [linear ramp 0 to 1] for 0<=t<=1,
        1 for 1<=t<=2,
        [linear ramp 1 to 2] for 1<=t<=2,
        2 for 2<=t<=3,
        [linear ramp 2 to 0] for 3<=t<=4
        and 0 otherwise
        """
        vals = [1, 2]
        times = [0, 1, 2, 3, 4, 5]
        ramp = parameter.RampParameter(vals, times)
        self.assertEqual(ramp.value(-1), 0)
        self.assertEqual(ramp.value(0), 0)
        self.assertEqual(ramp.value(0.5), 0.5)
        self.assertEqual(ramp.value(1), 1)
        self.assertEqual(ramp.value(1.2), 1)
        self.assertEqual(ramp.value(2), 1)
        self.assertAlmostEqual(ramp.value(2.5), 1.5)
        self.assertAlmostEqual(ramp.value(2.9), 1.9)
        self.assertEqual(ramp.value(3), 2)
        self.assertEqual(ramp.value(3.8), 2)
        self.assertEqual(ramp.value(4), 2)
        self.assertAlmostEqual(ramp.value(4.1), 1.8)
        self.assertAlmostEqual(ramp.value(4.5), 1)
        self.assertEqual(ramp.value(5), 0)
        self.assertEqual(ramp.value(6), 0)
