import torch

from .logistic import Logistic
from .transformed_distribution import TransformedDistribution


class LogLogistic(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        base_dist = Logistic(loc=loc, scale=scale, validate_args=validate_args)
        super().__init__(
            base_distribution=base_dist,
            transforms=[torch.distributions.ExpTransform()],
            validate_args=validate_args,
        )
