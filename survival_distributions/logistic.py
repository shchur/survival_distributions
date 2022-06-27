import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .survival_distribution import SurvivalDistribution


class Logistic(SurvivalDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def sf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return torch.sigmoid(-z)

    def logsf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return F.logsigmoid(-z)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return torch.sigmoid(z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return z - self.scale.log() - 2 * F.softplus(z)

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        z = (-u).log1p() - u.log()
        return self.scale * z + self.loc

    def _new_tensor(self, shape):
        return self.loc.new(shape)
