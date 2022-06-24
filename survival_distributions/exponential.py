import torch
from torch.distributions import constraints

from .survival_distribution import SurvivalDistribution


class Exponential(SurvivalDistribution, torch.distributions.Exponential):
    """Exponential distribution with rate parametrization.

    References:
        https://en.wikipedia.org/wiki/Exponential_distribution


    Args:
        rate: Rate parameter, equal to 1 / scale.
    """

    support = constraints.positive

    def logsf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -self.rate * value

    def log_hazard(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log().expand_as(value)

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        return -torch.log(u) / self.rate

    def icdf(self, u):
        return self.isf(1.0 - u)

    def _new_tensor(self, shape):
        return self.rate.new(shape)
