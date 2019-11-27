class AbstractCurrentProfile:
    """
    Abstract class describing the change of a current through time
    """

    def __init__(self):
        pass

    def current(self, t):
        """
        Return the current at the specified time.
        """
        raise NotImplementedError


class ConstantCurrent(AbstractCurrentProfile):
    """
    Profile of a current that is constant through time
    """

    def __init__(self, current):
        self._current = current

    def __eq__(self, other):
        if isinstance(other, ConstantCurrent):
            if self.current(0) == other.current(0):
                return True
        return False


    def current(self, t):
        return self._current
