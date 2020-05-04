"""
This file is part of Untitled Simulation Project

You can redistribute or modify it under the terms of the GNU General Public
License, either version 3 of the license or any later version.

Author: Cameron McGarry, 2020

Classes:

AbstractEvent: Abstract class to fomat later events
OutOfRangeSphere: Event describing loss outside of a sphere
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

    def event(self, t, state):
        """
        Check if the event has occurred. Should return 0 if so. To be
        implemented by subclasses

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        raise NotImplementedError

    def __call__(self, t, state, **kwargs):
        """
        Alias event method by with a call to the object for use with Desolver.
        Accept arbitrary kwargs for compatibility.

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        return self.event(t, state)

    @property
    def is_terminal(self):
        """
        Key-word attribute used by Desolver
        """
        return self._is_terminal

class OutOfRangeSphere(AbstractEvent):
    """
    Event determining if a particle has exceeded a given limit from the centre.
    The centre of the sphere translates through time according to a given
    paramter
    """

    def __init__(
            self,
            limit,
            center,
            terminal=True
            ):
        """
        Constructor for OutOfRangeSphereTranslate

        Args:
        limit: float, distance a particle can travel in any cardinal direction
        before terminating
        center: array parameter giving the cetner at a certain time
        terminal: bool, whether triggering the event should terminate the
        integration (default True)
        """
        self._limit = limit
        self._center = center

    def event(self, t, state, **kwargs):
        """
        Calling an event returns 0 if the particle has left the sphere

        Args:
        t: time
        state: Desolver state (anticipate 6-array of position and speed)
        unknown kwags
        """
        r_0 = self._center(t).copy()
        r = np.array(state[:3])
        if np.linalg.norm(r - r_0) > self._limit:
            return 0
        else:
            return 1
