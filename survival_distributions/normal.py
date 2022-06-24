import math

import torch

from .survival_distribution import SurvivalDistribution


class Normal(SurvivalDistribution, torch.distributions.Normal):
    def sf(self, value):
        z = (value - self.loc) / self.scale
        return 1.0 - torch.special.ndtr(z)

    def logsf(self, value):
        return torch.log(self.sf(value))

    def logcdf(self, value):
        z = (value - self.loc) / self.scale
        return torch.log(torch.special.ndtr(z))

    def isf(self, u):
        return self.loc + self.scale * torch.erfinv(1 - 2 * u) * math.sqrt(2)
