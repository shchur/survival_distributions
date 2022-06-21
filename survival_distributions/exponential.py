import torch
from numbers import Number


from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


from .survival_distribution import SurvivalDistribution


class Exponential(SurvivalDistribution):
    """Exponential distribution with rate parametrization.

    References:
        https://en.wikipedia.org/wiki/Exponential_distribution


    Args:
        rate: Rate of the distribution (lambda).
    """

    arg_constraints = {"rate": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, rate: torch.Tensor, validate_args=None):
        self.rate, = broadcast_all(rate)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = rate.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def logsf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -self.rate * value
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def log_hazard(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log().expand_as(value)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() * self.rate.reciprocal()

    def rsample_conditional(
        self, sample_shape=torch.Size(), lower_bound=None, upper_bound=None
    ):
        shape = self._extended_shape(sample_shape)
        u_min = self.cdf(lower_bound)
        u_max = self.cdf(upper_bound)
        # Convert Uniform[0, 1] sample to a Uniform[u_min, u_max] sample
        u = (u_max - u_min) * self.rate.new(shape).uniform_() + u_min
        return -u.log() * self.rate.reciprocal()

    def isf(self, u):
        if self._validate_args:
            assert constraints.unit_interval.check(u).all()
        return -torch.log(u) / self.rate

    def icdf(self, u):
        return self.isf(1.0 - u)

    def entropy(self):
        return 1.0 - self.rate.log()

    def _new_tensor(self, shape):
        return self.rate.new(shape)

    @property
    def mean(self):
        return self.rate.reciprocal()

    @property
    def variance(self):
        return self.rate.pow(-2)
