import numpy as np

from USP.wire import WireCluster
from USP.field import StaticField

from USP import parameter

class AbstractPotentialTrap:
    """
    AbstractPotentialTrap is a trap that takes a potential function supplied
    directly by the user. Useful for when a simple potential can be provided
    without the need for significant construction.
    """

    def __init__(self, potential_function):
        """
        Takes the potential function (with args r and t) that can be evaluated
        """
        # TODO Check that potential_function is of the correct form
        self.potential_function = potential_function

    def potential(self, r, t):
        """
        """
        return self.potential_function(r, t)

class AbstractTrap:

    def __init__(self):
        pass

    def field(self, r, t):
        """
        Return the trap field at position r and time t
        """
        raise NotImplementedError

    def potential(self, r, t):
        """
        Return the trap potential at position r and time t
        """
        return np.linalg.norm(self.field(r, t))

class FieldTrap(AbstractTrap):
    """
    A trap that uses field method defined externally (e.g. can take a
    QuadrupoleField.field)
    """

    def __init__(self, field):
        self.field = field

class ClusterTrapStatic(AbstractTrap):
    """
    Create a trap based on a wire cluster and a static bias field.
    """

    def __init__(self, cluster, bias):
        if not isinstance(cluster, WireCluster):
            raise ValueError(
                'ClusterTrap must have argument cluster of type WireCluster'
                )

        if not isinstance(bias, StaticField):
            raise ValueError(
                'ClusterTrap argument bias must be None or of type StaticField'
                )
        self.cluster = cluster
        self.bias = bias
        
    def field(self, t, r):
        """
        Return the trap field at position r and time t. Overrides superclass
        abstract method
        """
        return self.cluster.field(t, r) + self.bias.field(t, r)

class ClusterTrapStaticTR(ClusterTrapStatic):
    """
    Create a trap based on a wire cluster and a static bias field, with the
    latter generated to create a zero at specified time and position
    """

    def __init__(self, cluster, time, position):
        """
        """
        bias_field = StaticField(-1 * cluster.field(time, position))
        super().__init__(cluster, bias_field)


class ClusterTrap(AbstractTrap):
    """
    Create a trap based on a wire cluster and a height, which should be a
    paramater.AbstractParameterProfile subclass.

    TODO: Allow specification of other coordinates
    TODO: Allow bias to come from specified directions (e.g. y only)
    """

    def __init__(self, cluster, height):
        """
        """
        if not isinstance(cluster, WireCluster):
            raise ValueError(
                'ClusterTrap must have argument cluster of type WireCluster'
                )
        if not isinstance(height, parameter.AbstractParameterProfile):
            raise ValueError(
                'ClusterTrap must have argument height of type AbstractParameterProfile'
                )
        self._cluster = cluster
        self._height = height

    @property
    def cluster(self):
        return self._cluster

    def height(self, t):
        return self._height.value(t)

    def bias_field(self, t):
        """
        Bias field cancels the trapping field at specified height
        """
        r = [0, 0, self.height(t)]
        bias_field = -1 * self.cluster.field(t, r)
        return bias_field

    def field(self, t, r):
        """
        Return the trap field at position r and time t. Overrides superclass
        abstract method
        """
        return self.cluster.field(t, r) + self.bias_field(t)
