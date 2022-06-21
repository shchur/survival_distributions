from collections import namedtuple

import torch
from torch.distributions import constraints

from survival_distributions import Exponential, Weibull

Example = namedtuple("Example", ["Dist", "params"])

NUM_SAMPLES = 100_000
EXAMPLES = [
    Example(
        Exponential,
        [
            {"rate": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
            {"rate": torch.tensor([2.3], requires_grad=True)},
            {"rate": 2.3},
        ],
    ),
    Example(
        Weibull,
        [
            {
                "rate": torch.tensor([2.0, 0.5, 1.1]),
                "concentration": torch.tensor([0.5, 2.5, 1.0]),
            },
            {"rate": torch.tensor([1.7]), "concentration": torch.tensor([0.1])},
            {"rate": 4.0, "concentration": 2.3},
        ],
    ),
]


def grid_on_support(dist, num_grid_points=500, eps=1e-20):
    if dist.support == constraints.positive:
        grid = torch.linspace(eps, 20, num_grid_points)
    elif dist.support == constraints.real:
        grid = torch.linspace(-20, 20, num_grid_points)
    else:
        raise ValueError("dist must have support positive or real")
    # The grid must have shape (num_grid_points,) + dist.batch_shape
    expanded_shape = grid.shape + (1,) * len(dist.batch_shape)
    return grid.view(expanded_shape).expand(grid.shape + dist.batch_shape)


def test_ecdf(tolerance=1e-2):
    """Check whether the empirical CDF based on the samples is close to the true CDF."""
    for Dist, configs in EXAMPLES:
        for config in configs:
            dist = Dist(**config)
            grid = grid_on_support(dist)
            # Compute true CDF on the grid points
            cdf = dist.cdf(grid)
            # Compute empirical CDF on the grid points
            samples = dist.sample([NUM_SAMPLES])
            assert (
                samples.shape
                == torch.Size([NUM_SAMPLES]) + dist.batch_shape + dist.event_shape
            )
            ecdf = (samples.unsqueeze(1) < grid).float().mean(0)
            return torch.allclose(cdf, ecdf, atol=tolerance)


def test_sf(tolerance=1e-4):
    """Test whether cdf, sf, log_cdf and log_sf return compatible values."""
    for Dist, configs in EXAMPLES:
        for config in configs:
            dist = Dist(**config)
            grid = grid_on_support(dist)
            cdf = dist.cdf(grid)
            sf = dist.sf(grid)
            log_sf = dist.logsf(grid)
            log_cdf = dist.logcdf(grid)
            assert torch.allclose(cdf, 1 - sf, atol=tolerance)
            assert torch.allclose(cdf, 1 - log_sf.exp(), atol=tolerance)
            assert torch.allclose(cdf, log_cdf.exp(), atol=tolerance)

            log_pdf = dist.log_prob(grid)
            log_hazard = dist.log_hazard(grid)
            assert torch.allclose(log_pdf - log_sf, log_hazard, atol=tolerance)


def test_density(tolerance=1e-5):
    """Compare two ways for computing the PDF of the distribution.

    1. Using the log_prob method.
    2. Derivative of the survival function: p(x) = -d/dx S(x).
    """
    for Dist, configs in EXAMPLES:
        for config in configs:
            dist = Dist(**config)
            x = grid_on_support(dist)
            x = x.requires_grad_()
            sf = dist.sf(x)
            sf.sum().backward()
            pdf = -x.grad
            assert torch.allclose(pdf, dist.log_prob(x.detach()).exp(), atol=tolerance)
