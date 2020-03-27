"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Classes:

StaticField: Provides field method for an arbitrary constant field
QuadrupoleField: Provides field method for a quadrupolar field
QuadrupoleFieldTranslate: Similar to the above but with ability to move the trap
centre as time progresses
"""

import numpy as np

from . import utils
from . import parameter

class StaticField:
    """
    Return a field of constant value
    """

    def __init__(self, field_array):
        """
        Constructor for StaticField

        Args:
        field_array: list-like, the components of the static field in x, y and z
        """
        self.field_array = np.array(field_array)

    def field(self, t, r):
        """
        Returns the field value at event t, r

        Args:
        t: float, time
        r: list-like, spatial position
        """
        return self.field_array
        
class QuadrupoleField:
    """
    Approximation of a quadrupole field close to the centre
    """

    def __init__(self, b_1, r_0=[0,0,0]):
        """
        Constructor for QuadrupoleField

        Args:
        b_1: USP.parameter, field gradient
        r_0: list-like, zero position (optional, default [0,0,0])
        """
        self.b_1 = b_1
        self.r_0 = utils.clean_vector(r_0, cast_type=float)

    def field(self, t, r):
        """
        Returns the field value at event t, r

        Args:
        t: float, time
        r: list-like, spatial position
        """
        r = np.array(r) - self.r_0
        scale = np.array([-0.5, -0.5, 1])
        return self.b_1(t) * scale * r

class QuadrupoleFieldTranslate(QuadrupoleField):
    """
    Quadrupole field translating in time. Move in a cardinal direction depending
    according to some parameter
    """

    def __init__(
            self,
            b_1,
            z_param,
            r_0=[0,0,0],
            direction=2
            ):
        """
        Constructor for QuadrupoleFieldTranslate

        Args:
        b_1: float, field gradient
        z_param: USP.parameter object, determines position along `direction` axis
        r_0: list-like, zero position (optional, default [0,0,0])
        direction: str or int, represents the (cardinal) direction of
        translation
        """
        super().__init__(b_1, r_0)
        # FIXME importing USP.parameter breaks multiprocessing
        #if not isinstance(param, parameter.AbstractParameterProfile):
        #    raise ValueError('QuadrupoleFieldTranslate expects a USP parameter')
        self.z_param = z_param
        self.direction = utils.clean_direction_index(direction)

    def field(self, t, r):
        """
        Returns the field value at event t, r

        Args:
        t: float, time
        r: list-like, spatial position
        """
        r_0 = self.r_0.copy()
        r_0[self.direction] = self.z_param(t)
        r = np.array(r) - r_0
        scale = np.array([-0.5, -0.5, 1.])
        return self.b_1(t) * scale * r
