import numpy as np
import scipy.constants as spc

from USP.parameter import AbstractParameterProfile
from . import utils

class WireSegment:
    """
    Simulates a segment of straight wire. Initialise with a start and end (3
    vectors) and current (scalar)
    """

    def __init__(self, current, start, end):
        self.set_start(start)
        self.set_end(end)
        self.set_current(current)

    def __eq__(self, other):
        """
        Override built in method. If start, end and current are the same then
        return True. Otherwise return False
        """
        if isinstance(other, WireSegment):
            if self.current == other.current:
                if np.array_equal(self.start, other.start):
                    if np.array_equal(self.end, other.end):
                        return True
        return False

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def current(self):
        return self._current

    @property
    def wire_vector(self):
        """
        Wire as a vector
        """
        return self.end - self.start

    def _clean_vector(self, vector, length=3):
        """
        Ensure a vector is of the right length and format before setting it
        """
        return utils.clean_vector(vector, length)

    def set_start(self, start):
        self._start = self._clean_vector(start)

    def set_end(self, end):
        self._end = self._clean_vector(end)

    def set_current(self, current):
        if not isinstance(current, AbstractParameterProfile):
            warn = 'Current must inherit from current.AbstractParameterProfile'
            raise ValueError(warn)
        else:
            self._current = current

    def field(self, t, r):
        """
        Return the field generated by the wire at time t and position r
        """
        r = self._clean_vector(r)
        current = self.current.value(t)
        # Find the smallest vector from r to wire, if the wire were inifinite
        param = np.dot(r - self.start, self.wire_vector)
        param /= np.dot(self.wire_vector, self.wire_vector) 
        a = self.start - r + param * self.wire_vector
        a_norm = np.linalg.norm(a)
        # Field direction is normal to the wire-point plane
        start_rel = r - self.start
        direction = np.cross(self.wire_vector, start_rel)
        direction = direction / np.linalg.norm(direction)
        # Magnitude from the standard formula
        cos_start = np.dot(start_rel, a)
        cos_start /= (np.linalg.norm(start_rel) * a_norm)
        # Avoid floating point errors by checking here if cos_start is too big
        if cos_start**2 > 1:
            cos_start = 1
        sin_start = np.sqrt(1 - cos_start**2)
        end_rel = r - self.end
        cos_end = np.dot(end_rel, a)
        cos_end /= (np.linalg.norm(end_rel) * a_norm)
        if cos_end**2 > 1:
            cos_end = 1
        sin_end = np.sqrt(1 - cos_end**2)
        mag = spc.mu_0 * current * (sin_end + sin_start) / (4*np.pi*a_norm)
        return mag * direction


class WireCluster:
    """
    Holds a list of WireSegments. Can be used to find the fields of all of them
    by taking the sum of each segment's contribution.
    """
    def __init__(self, wires):
        self.set_wires(wires)

    def __eq__(self, other):
        """
        Override built in equality check. If two WireClusters have all equal
        wires then return True, otherwise return False
        """
        if isinstance(other, WireCluster):
            if self.length == other.length:
                for _ in range(self.length):
                    if self.wires[_] != other.wires[_]:
                        return False
                return True
        return False

    @property
    def wires(self):
        return self._wires

    @property
    def length(self):
        return len(self.wires)

    def set_wires(self, wires):
        """
        Check that each element of the list is a WireSegment
        """
        for wire in wires:
            if type(wire) != WireSegment:
                raise ValueError(
                    'All wires in cluster must have type WireSegment'
                    )
        self._wires = wires.copy()

    def field(self, t, r):
        """
        Get cluster field by summing each segment's contribution
        """
        field = [ wire.field(t, r) for wire in self._wires ]
        return np.array(sum(field))


class ZWire(WireCluster):
    """
    A WireCluster forming a single Z wire trap
    """

    def __init__(self, current, axial_length, end_length=5e2):
        axial_length = float(axial_length)
        end_length = float(end_length)
        wires = self.create_wires(current, axial_length, end_length)
        self.set_wires(wires)
        # TODO Orientation
        # TODO Position (including height)

    def create_wires(self, current, axial_length, end_length):
        al = axial_length
        el = end_length
        end_left = WireSegment(current, [-el, -al/2, 0], [0, -al/2, 0])
        axis = WireSegment(current, [0, -al/2, 0], [0, al/2, 0])
        end_right = WireSegment(current, [0, al/2, 0], [el , al/2, 0])
        return [end_left, axis, end_right]

class UWire(ZWire):
    """
    A WireCluster forming a single U wire trap
    """

    def create_wires(self, current, axial_length, end_length):
        """
        Override ZWire create_wires method to form a UWire
        """
        al = axial_length
        el = end_length
        end_left = WireSegment(current, [-al/2, -el, 0], [-al/2, 0, 0])
        axis = WireSegment(current, [-al/2, 0, 0], [al/2, 0, 0])
        end_right = WireSegment(current, [al/2, 0, 0], [al/2, -el, 0])
        return [end_left, axis, end_right]
