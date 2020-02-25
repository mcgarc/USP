"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Classes:

AbstractEvent: Abstract class to fomat later events
OutOfRangeBox: Event describing loss by a limit along each cardinal direction
"""

import numpy as np

from . import utils


class AbstractEvent:
    """
    Abstracted event to be passed to Desolver integrator using the "events"
    keyword option. Has attribute is_terminal to signify if triggering the
    event terminates the integration. __call__ must be implemented by
    subclasses.
    """

    def __init__(self, terminal=True):
        """
        Constructor for AbstractEvent

        Args:
        terminal: bool, whether triggering the event should terminate the
        integration (default True)
        """
        self._is_terminal = terminal

    def __call__(self, t, state, **kwargs):
        """
        Calling an event returns 0 if the event has taken place.

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        raise NotImplementedError

    @property
    def is_terminal(self):
        """
        Key-word attribute used by Desolver
        """
        return self._is_terminal


class OutOfRangeBox(AbstractEvent):
    """
    Event determining if a particle has exceeded a given limit in any of the
    cardinal directions. i.e. If it has left the cube surrounding a specified
    centre point, with edge length 2*limit
    """

    def __init__(self, limit, center=[0, 0, 0], terminal=True):
        """
        Constructor for OutOfRangeBox

        Args:
        limit: float, distance a particle can travel in any cardinal direction
        before terminating
        center: the position from which to evaluate this distance
        terminal: bool, whether triggering the event should terminate the
        integration (default True)
        """
        super().__init__(terminal)
        self._limit = limit
        self._center = np.array(center).astype(float)

    def __call__(self, t, state, **kwargs):
        """
        Calling an event returns 0 if the particle has left the box

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        r = state[:3]
        v = state[3:]
        if any(abs(r - self._center) > self._limit):
            return 0
        else:
            return 1

class OutOfRangeSphere(OutOfRangeBox):
    """
    Event determining if a particle has exceeded a given limit from the centre.
    i.e. If it has left the sphere sourrounding the centre with radius limit
    """

    def __call__(self, t, state, **kwargs):
        """
        Calling an event returns 0 if the particle has left the sphere

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        r = np.array(state[:3])
        if np.linalg.norm(r - self._center) > self._limit:
            return 0
        else:
            return 1

class OutOfRangeSphereTranslate(OutOfRangeBox):
    """
    Event determining if a particle has exceeded a given limit from the centre.
    The centre of the sphere translates through time according to a given
    paramter
    """

    def __init__(
            self,
            limit,
            param,
            direction=0,
            center=[0, 0, 0],
            terminal=True
            ):
        """
        Constructor for OutOfRangeSphereTranslate

        Args:
        limit: float, distance a particle can travel in any cardinal direction
        before terminating
        param: USP.parameter object, determines position along `direction` axis
        direction: str or int, represents the (cardinal) direction of
        translation
        center: the position from which to evaluate this distance
        terminal: bool, whether triggering the event should terminate the
        integration (default True)
        """
        super().__init__(limit, center, terminal)
        self._direction = utils.clean_direction_index(direction)
        self._param = param

    def __call__(self, t, state, **kwargs):
        """
        Calling an event returns 0 if the particle has left the sphere

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        r_0 = self._center.copy()
        r_0[self._direction] = self._param.value(t)
        r = np.array(state[:3])
            #print(r, r_0, r-r_0, self._param.value(t))
        if np.linalg.norm(r - r_0) > self._limit:
            return 0
        else:
            return 1
