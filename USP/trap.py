import numpy as np

from USP.wire import WireCluster
from USP.field import StaticField

from USP import parameter
from USP import consts
from USP import utils
from USP import figs

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

    def potential(self, t, r):
        """
        """
        return self.potential_function(t, r)

class AbstractTrap:

    def __init__(self):
        pass

    def field(self, t, r):
        """
        Return the trap field at position r and time t
        """
        raise NotImplementedError

    def potential(self, t, r):
        """
        Return the trap potential at position r and time t
        """
        return np.linalg.norm(self.field(t, r))

    def plot_potential_cut(
            self,
            t,
            x_start,
            x_end,
            direction,
            points=100,
            cut_point=[0, 0, 0],
            output_path=None,
            figsize=figs.DEFAULT_FIGSIZE,
            dpi=figs.DEFAULT_DPI
            ):
        """
        Plot a cut through of the potential along specified direction.
        """
        r = utils.clean_vector(cut_point)
        direction, dir_label = utils.clean_direction_index(
                direction,
                str_rep=True
                )
        x_data = np.linspace(x_start, x_end, num=points)
        y_data = []
        for point in x_data:
            r[direction] = point
            y_data.append(self.potential(t, r))
        y_data = np.array(y_data)
        utils.plot_2D_line(
            x_data,
            y_data,
            f'Trap potential cut through {dir_label}',
            f'{dir_label} (m)',
            'potential (J)',
            figsize,
            dpi,
            output_path
            )

class FieldTrap(AbstractTrap):
    """
    A trap that uses field method defined externally (e.g. can take a
    QuadrupoleField.field)
    """

    def __init__(self, field):
        self.field = field

class ZeemanGuide(AbstractPotentialTrap):
    """
    Zeeman guiding potential
    """

    def __init__(
            self,
            B0str = 1.05,
            B1strX = 1.25,
            B0strY = 1.0,
            B0weak = 0.05,
            B1weakX = 0.3,
            B1weakY = 0.5,
            B2weakX = 0.04,
            B2weakY = 0.05,
            L = 2E-2,
            R = 2.5E-3
            ):
        """
        I took the necessary formulae from Gautam's Mathematica notebook,
        so I don't really understand what any of these mean.
        """
        self.B0str = B0str 
        self.B1strX = B1strX
        self.B0strY = B0strY
        self.B0weak = B0weak
        self.B1weakX = B1weakX
        self.B1weakY = B1weakY
        self.B2weakX = B2weakX
        self.B2weakY = B2weakY
        self.L = L
        self.R = R


    def axial_B_sin(self, z, L):
        z = float(z)
        L = float(L)
        return np.sin(np.pi*z/L)**2
    
    def radial_strong_B(self, rho, B0, B1, R):
        rho = float(rho)
        B0 = float(B0)
        B1 = float(B1)
        R = float(R)
        return (B1 - B0)*rho**2/R**2
    
    def radial_weak_B(self, rho, B0, B1, B2, R):
        rho = float(rho)
        B0 = float(B0)
        B1 = float(B1)
        B2 = float(B2)
        R = float(R)
        # a soln
        det_a = B0*B1 - B0*B2 - B1*B2 + B2**2
        # ignore very small values (floating point happens)
        if det_a < 1E-10:
            det_a = 0
        a = ((B0 + B1 - 2*B2)*R**4 + 2*np.sqrt(det_a)*R**4)/R**8
        # b soln
        det_b = (B1-B2)*(B0-B2)
        if det_b < 0:
            print(B0, B1, B2, det_b)
        b = 2*(B2 - B0 - np.sqrt(det_b))/R**2
        return a*rho**4 + b*rho**2
    
    
    def potential(self, t, r):
        x = r[0]
        y = r[1]
        z = r[2]
        B0str = 1.05
        pot = self.axial_B_sin(z, self.L) * (
                self.radial_strong_B(x, self.B0str, self.B1strX, self.R) +
                self.radial_strong_B(y, self.B0str, self.B0strY, self.R) +
                self.B0str
                ) + self.axial_B_sin(z + self.L/2, self.L) * (
                self.radial_weak_B(
                    x,
                    self.B0weak,
                    self.B1weakX,
                    self.B2weakX,
                    self.R
                    ) +
                self.radial_weak_B(
                    y,
                    self.B0weak,
                    self.B1weakY,
                    self.B2weakY,
                    self.R
                    ) +
                self.B0weak
                )
        return pot * consts.u_B

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
    """

    def __init__(self, cluster, height, bias_scale=[0, 0, 0]):
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
        self._bias_scale = np.array(bias_scale)

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
        return bias_field * self._bias_scale

    def field(self, t, r):
        """
        Return the trap field at position r and time t. Overrides superclass
        abstract method
        """
        return self.cluster.field(t, r) + self.bias_field(t)

class SuperimposeTrap(AbstractTrap):
    """
    """

    def __init__(self, traps):
        """
        Take the list of all the traps to sum over
        """
        self._traps = traps

    def field(self, t, r):
        """
        Sum fields for all traps
        """
        field_comps = [trap.field(t, r) for trap in self._traps]
        return np.array(sum(field_comps))
