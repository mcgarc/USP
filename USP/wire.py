import numpy as np
import scipy.constants as spc

from USP.parameter import AbstractParameterProfile, ConstantParameter
from USP import consts
from . import utils

def clean_current(current):
    """
    If the current is already a parameter then set it, otherwise attempt to
    convert a float-castable to constant parameter.
    """
    if isinstance(current, AbstractParameterProfile):
        return current
    else:
        try:
            current = float(current)
        except ValueError:
            warn = 'Current must be parameter or castable to float'
            raise ValueError(warn)
        current = ConstantParameter(current)
        return current

class WireInfinite:
    """
    Simulates an unphysical wire of infinite length. Superclass for
    WireInfinite[Direction]
    """
    def __init__(self, current, position, direction=None):
        self.set_current(current)
        self._position = utils.clean_vector(position, 3) # TODO setter
        if direction is not None:
            raise NotImplemented
            # TODO Generalised direction that doesn't slow simple versions
        else:
            self._direction = None

    def set_current(self, current):
        self._current = clean_current(current)

    def field(self, t, r):
        """
        Infinite wire fields where direction is not specified here should be
        implemented by a subclass. Otherwise calculate the field directly
        """
        if self._direction is None:
            # Expect a direction or an overloaded field from subclass
            raise NotImplementedError
        else:
            raise NotImplemented # TODO

class WireInfiniteX(WireInfinite):
    """
    Infinite wire parallel to X
    """

    def __init__(self, current, position):
        super().__init__(current, position, None)
        self._position[0] = 0

    def field(self, t, r):
        current = self._current.value(t)
        # Vector from wire to point
        r[0] = 0
        r_prime = r - self._position
        r_prime_sqd = np.dot(r_prime, r_prime)
        # Field direction
        field_direction = np.array([0, -r_prime[2], r_prime[1]])
        field_mag_norm = consts.u_0 * current / (2*np.pi*r_prime_sqd)
        return  field_mag_norm * field_direction

class WireInfiniteY(WireInfinite):
    """
    Infinite wire parallel to Y
    """

    def __init__(self, current, position):
        super().__init__(current, position, None)
        self._position[1] = 0

    def field(self, t, r):
        current = self._current.value(t)
        # Vector from wire to point
        r[1] = 0
        r_prime = r - self._position
        r_prime_sqd = np.dot(r_prime, r_prime)
        # Field direction
        field_direction = np.array([-r_prime[2], 0, r_prime[0]])
        field_mag_norm = consts.u_0 * current / (2*np.pi*r_prime_sqd)
        return  field_mag_norm * field_direction

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
        self._current = clean_current(current)

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

    def __init__(self, current, axial_length, end_length=5e2, center=[0,0,0]):
        axial_length = float(axial_length)
        end_length = float(end_length)
        self._center = utils.clean_vector(center)
        wires = self.create_wires(current, axial_length, end_length)
        self.set_wires(wires)
        # TODO Orientation

    def create_wires(self, current, axial_length, end_length):
        al = axial_length
        el = end_length
        x, y, h = self._center
        end_left = WireSegment(current, [x-el, y-al/2, h], [x, y-al/2, h])
        axis = WireSegment(current, [x, y-al/2, h], [x, y+al/2, h])
        end_right = WireSegment(current, [x, y+al/2, h], [x+el, y+al/2, h])
        return [end_left, axis, end_right]

class ZWireXAxis(ZWire):
    """
    ZWire but oriented with axis along x direction
    TODO: Replace with orientation for ZWire
    """

    def create_wires(self, current, axial_length, end_length):
        al = axial_length
        el = end_length
        x, y, h = self._center
        end_left = WireSegment(current, [x-al/2, y-el, h], [x-al/2, y, h])
        axis = WireSegment(current, [x-al/2, y, h], [x+al/2, y, h])
        end_right = WireSegment(current, [x+al/2, y, h], [x+al/2, y+el, h])
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
        x, y, h = self._center
        end_left = WireSegment(current, [x-al/2, y-el, h], [x-al/2, y, h])
        axis = WireSegment(current, [x-al/2, y, h], [x+al/2, y, h])
        end_right = WireSegment(current, [x+al/2, y, h], [x+al/2, y-el, h])
        return [end_left, axis, end_right]
