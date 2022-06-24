import torch

from .survival_distribution import SurvivalDistribution


class TransformedDistribution(
    torch.distributions.TransformedDistribution, SurvivalDistribution
):
    base_dist: SurvivalDistribution

    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=validate_args)
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if sign not in (-1, 1):
            raise ValueError("Each transform must have sign +1 or -1")
        # sign equals to -1 if the transformation is decreasing, +1 if increasing
        self.sign = int(sign)

    def logsf(self, value):
        x = value
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)
        if self.sign == 1:
            return self.base_dist.logsf(x)
        else:
            return self.base_dist.logcdf(x)

    def logcdf(self, value):
        x = value
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)
        if self.sign == 1:
            return self.base_dist.logcdf(x)
        else:
            return self.base_dist.logsf(x)

    def log_hazard(self, value):
        # The naive implementation `log_prob(value) - logsf(value)` goes through all the
        # transformations twice. This implementation only goes through the transforms
        # once
        log_hazard = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_hazard = log_hazard - transform.log_abs_det_jacobian(x, y)
            y = x

        log_hazard = log_hazard + self.base_dist.log_hazard(y)
        return log_hazard

    def isf(self, u):
        x = self.base_dist.isf(u)
        for transform in self.transforms:
            x = transform(x)
        return x

    def _new_tensor(self, shape):
        return self.base_dist._new_tensor(shape)
