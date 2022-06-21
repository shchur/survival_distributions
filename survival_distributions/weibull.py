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

    def __init__(
        self, rate: torch.Tensor, concentration: torch.Tensor, validate_args=None
    ):
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

    # def rsample(self, sample_shape=torch.Size()):
    #     shape = torch.Size(sample_shape) + self.batch_shape
    #     z = torch.empty(
    #         shape, device=self.rate.device, dtype=self.rate.dtype
    #     ).exponential_(1.0)
    #     samples = (z * self.rate.reciprocal()).pow(self.concentration.reciprocal())
    #     return samples

    # def rsample_conditional(
    #     self, sample_shape=torch.Size(), lower_bound=None, upper_bound=None
    # ):
    #     shape = self._extended_shape(sample_shape)
    #     u_min = self.cdf(lower_bound)
    #     u_max = self.cdf(upper_bound)
    #     # Convert Uniform[0, 1] sample to a Uniform[u_min, u_max] sample
    #     u = (u_max - u_min) * self.rate.new(shape).uniform_() + u_min
    #     return (-u.log() * self.rate.reciprocal()).pow(self.concentration.reciprocal())
