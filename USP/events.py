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
        self._center = np.array(center)

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
