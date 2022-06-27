import torch
from torch.distributions import constraints

from .survival_distribution import SurvivalDistribution


class Uniform(torch.distributions.Uniform, SurvivalDistribution):
    def logsf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self.high - value).log() - (self.high - self.low).log()

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        result = self.high - u * (self.high - self.low)
        return result

    def _new_tensor(self, shape):
        return self.low.new(shape)
