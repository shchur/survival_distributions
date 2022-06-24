from numbers import Number
from typing import Union

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .survival_distribution import SurvivalDistribution


class Weibull(SurvivalDistribution):
    arg_constraints = {
        "rate": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, rate, concentration, validate_args=None):
        self.rate, self.concentration = broadcast_all(rate, concentration)
        batch_shape = self.rate.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def logsf(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return self.rate.neg() * torch.pow(x, self.concentration)

    def log_prob(self, x):
        return self.log_hazard(x) + self.logsf(x)

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        return (-u.log() * self.rate.reciprocal()).pow(self.concentration.reciprocal())

    def log_hazard(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return (
            self.rate.log()
            + self.concentration.log()
            + (self.concentration - 1) * x.log()
        )

    @property
    def mean(self):
        log_lmbd = self.concentration.reciprocal().neg() * self.rate.log()
        return torch.exp(log_lmbd + torch.lgamma(1 + self.concentration.reciprocal()))

    def _new_tensor(self, shape):
        return self.rate.new(shape)
