"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Functions:
clean_parameter
clean_array_parameter

Classes:

AbstractParameterProfile
ArrayParameter
ConstantParameter
StepParameter
RampParameter
"""

import warnings
import numpy as np

def clean_parameter(param):
    """
    Takes an argument and ensures it is a USP parameter. Constants are
    converted to ConstantParameter
    """
    if np.isscalar(param):
        param = ConstantParameter(param)
    elif not isinstance(param, AbstractParameterProfile):
        raise ValueError('Expected a parameter, or something castable \
                to a constant parameter.')
    return param


def clean_array_parameter(array):
    """
    Takes an array-like or array parameter and returns an ArrayParameter
    object.
    """
    if isinstance(array, ArrayParameter):
        return array
    else:
        return ArrayParameter(array)


class AbstractParameterProfile:
    """
    Abstract class describing the change of a scalar parameter through time
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        """
        Should override built in equality method
        """
        warnings.warn(
                'Parameter profile uses built-in equality method',
                UserWarning
                )
        return super().__eq__(other)

    def value(self, t):
        """
        Return the parameter's value at the specified time.
        """
        raise NotImplementedError

    def __call__(self, t):
        """
        Alias call as self.value
        """
        return self.value(t)

class ArrayParameter(AbstractParameterProfile):
    """
    Parameter that returns an array rather than float or int. Takes a list of
    parameters (or 0) to construct.
    """

    def __init__(self, param_list):
        """
        Take a list of parameters or constants and save as a list of parameter
        types
        """
        self._params = []
        for param in param_list:
            param = clean_parameter(param)
            self._params.append(param)

    def value(self, t):
        """
        Get the parameter value at t for each member of params list
        """
        return [param(t) for param in self._params]

                
class ConstantParameter(AbstractParameterProfile):
    """
    Profile of a parameter that is constant through time
    """

    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        if isinstance(other, ConstantParameter):
            if self.value(0) == other.value(0):
                return True
        return False


    def value(self, t):
        return self._value

class StepParameter(AbstractParameterProfile):
    """
    Parameter that steps through time. Takes a list of parameter values and a
    corresponding list of times of the same length. Each parameter element is
    its value *up to and excluding* the corresponding time element
    """

    def __init__(self, values, times):
        self._values = values
        self._times = sorted(times)
        self._size = len(values)
        self._check_times_values()

    def __eq__(self, other):
        instance = isinstance(other, StepParameter)
        size = self._size == len(other)
        vals = self.values == other.values
        times = self.times == other.times
        if instance and size and vals and times:
            return True
        return False

    def __len__(self):
        return self._size

    def _check_times_values(self):
        """
        Check the relative length of values and times
        """
        if len(self.values) + 1 != len(self.times):
            raise ValueError('Mismatched values and time lengths')

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values
        
    def value(self, t):
        cond_list = [t < self.times[0]]
        cond_list += [self.times[i] <= t and t < self.times[i+1] for i in range(self._size)] 
        cond_list += [t >= self.times[-1]]
        val_list = [0] + self.values + [0]
        return float(np.piecewise(t, cond_list, val_list))

class RampParameter(StepParameter):
    """
    Parameter that ramps through time. Takes a list of values and a
    corresponding list of times. At each time, ramp up to the next values and
    hold at that value until the next time. Ramps are linear.

    For inputs of values = [1] and times = [0, 1, 2, 3] ramp will look like 
   1|     _____
    |    /     \
    |   /       \
    |  /         \
    | /           \
    |/             \
   0------------------
    0    1    2     3
    """

    def __init__(self, values, times):
        super().__init__(values, times)

    def _check_times_values(self):
        """
        Overrides superclass check for relative length of times and values 
        """
        if len(self.times) != 2*len(self.values) + 2:
            raise ValueError('Mismatched values and time lengths')
        if len(self.times) <= 6:
            warnings.warn(
            'This class is inefficient, consider using a specific ramp class.'
            )

    @staticmethod
    def value_ramp(v_i, v_f, t_i, t_f, t):
        """
        Returns the value of the ramp between c_i and c_f over t_i and t_f at
        time t
        """
        grad = (v_f - v_i) / (t_f - t_i)
        return grad * (t - t_i) + v_i

    def value(self, t):
        """
        """
        #return self.values[0]
        cond_list = [self.times[i] <= t and t <= self.times[i+1] for i in range(len(self._times) - 1)] 
        cond_list = [t < self.times[0]] + cond_list + [t > self.times[-1]]
        values = [0] + self.values + [0]
        region_index = cond_list.index(True)
        if region_index % 2:
            # Ramping phase
            return self.value_ramp(
                    values[int((region_index - 1)/2)],
                    values[int((region_index + 1)/2)],
                    self.times[region_index - 1],
                    self.times[region_index],
                    t
                    )
        else:
            # Constant phase
            return values[int(region_index/2)]

class RampParameterUpDown(RampParameter):
    """
    Simplified version of ramp parameter with four times and two values, do do
    just one ramp up to a value and then back down.
    """

    def __init__(self, t1, t2, t3, t4, v2, v1=0):
        """
        Takes the four critical times, the second value and the first value
        (defaults to 0).
        """
        self._t1 = t1
        self._t2 = t2
        self._t3 = t3
        self._t4 = t4
        self._v1 = v1
        self._v2 = v2

    def value(self, t):
        if t <= self._t1:
            return self._v1
        elif t <= self._t2:
            return self.value_ramp(self._v1, self._v2, self._t1, self._t2, t)
        elif t <= self._t3:
            return self._v2
        elif t <= self._t4:
            return self.value_ramp(self._v2, self._v1, self._t3, self._t4, t)
        else:
            return self._v1

class RampParameterUpDown2(RampParameter):
    """
    Simplified version of ramp parameter with three ramps
    """

    def __init__(self, t1, t2, t3, t4, t5, t6, v2, v3, v1=0):
        """
        Takes the six critical times, the second and third values, and the
        first value (defaults to 0).
        """
        self._t1 = t1
        self._t2 = t2
        self._t3 = t3
        self._t4 = t4
        self._t5 = t5
        self._t6 = t6
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3

    def value(self, t):
        if t <= self._t1:
            return self._v1
        elif t <= self._t2:
            return self.value_ramp(self._v1, self._v2, self._t1, self._t2, t)
        elif t <= self._t3:
            return self._v2
        elif t <= self._t4:
            return self.value_ramp(self._v2, self._v3, self._t3, self._t4, t)
        elif t <=self._t5:
            return self._v3
        elif t <= self._t6:
            return self.value_ramp(self._v3, self._v1, self._t5, self._t6, t)
        else:
            return self._v1

class SigmoidParameter(AbstractParameterProfile):
    """
    Simple ramp of parameter between values following a sigmoid.
    """

    def __init__(
            self,
            p_0,
            p_1,
            t_0,
            t_1,
            v,
            end_cap=False
            ):
        """
        SigmoidParameter constructor

        Args:
        p_0: float, initial value of parameter
        p_1: float, final value of parameter
        t_0: float, ramp start time
        t_1: float, ramp end time
        v: float, ramp speed
        end_cap: bool, if true, forces exact values of p_0 and p_1 for t < t_0
        and t > t_1 respectively
        """
        self.p_0 = float(p_0)
        self.p_1 = float(p_1)
        self.t_0 = float(t_0)
        self.t_1 = float(t_1)
        self.v = float(v)
        self.end_cap = end_cap

    def value(self, t):
        """
        Return the value of the Parameter at a given time

        Args:
        t: float, time of evaluation
        """
        if self.end_cap and t < self.t_0:
            return self.p_0
        elif self.end_cap and t > self.t_1:
            return self.p_1
        else:
            denom = 1 + np.exp(-self.v * (t - (self.t_0 + self.t_1)/2))
            p = self.p_0 + (self.p_1 - self.p_0) / denom
            return p
