import math

import torch
from torch.distributions import constraints

from .survival_distribution import SurvivalDistribution


class Normal(torch.distributions.Normal, SurvivalDistribution):
    def sf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return 1.0 - torch.special.ndtr(z)

    def logsf(self, value):
        return torch.log(self.sf(value))

    def logcdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return torch.log(torch.special.ndtr(z))

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        return self.loc + self.scale * torch.special.ndtri(1.0 - u)

    def _new_tensor(self, shape):
        return self.loc.new(shape)
