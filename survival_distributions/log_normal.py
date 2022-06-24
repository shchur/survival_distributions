import torch

from .normal import Normal
from .transformed_distribution import TransformedDistribution


class LogNormal(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc=loc, scale=scale, validate_args=validate_args)
        super().__init__(
            base_distribution=base_dist,
            transforms=[torch.distributions.ExpTransform()],
            validate_args=validate_args,
        )
