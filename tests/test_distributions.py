import pytest
import torch
from torch.distributions import constraints

from survival_distributions import Exponential, Normal, Weibull

NUM_SAMPLES = 100_000
EXAMPLES = [
    Exponential(rate=torch.tensor([0.7, 0.2, 0.4], requires_grad=True)),
    Exponential(rate=torch.tensor([2.3], requires_grad=True)),
    Exponential(rate=2.3),
    Normal(loc=torch.tensor([-1.5, 2.5, -5.0]), scale=torch.tensor([0.5, 2.5, 0.1])),
    Normal(loc=torch.tensor([-2.3]), scale=torch.tensor([0.1])),
    Normal(loc=4.1, scale=2.2),
    Weibull(
        rate=torch.tensor([2.0, 0.5, 1.1]), concentration=torch.tensor([0.5, 2.5, 1.0])
    ),
    Weibull(rate=torch.tensor([1.7]), concentration=torch.tensor([0.1])),
    Weibull(rate=4.0, concentration=2.3),
]


def grid_on_support(dist, num_grid_points=500, eps=1e-20, x_max=20):
    """Generate a grid on the support of the distribution"""
    if dist.support == constraints.positive:
        grid = torch.linspace(eps, x_max, num_grid_points)
    elif dist.support == constraints.real:
        grid = torch.linspace(-x_max, x_max, num_grid_points)
    else:
        raise ValueError("dist must have support positive or real")
    # The grid must have shape (num_grid_points,) + dist.batch_shape
    expanded_shape = grid.shape + (1,) * len(dist.batch_shape)
    return grid.view(expanded_shape).expand(grid.shape + dist.batch_shape)


@pytest.mark.parametrize("dist", EXAMPLES)
def test_ecdf(dist, tolerance=1e-2):
    """Check whether the empirical CDF based on the samples is close to the true CDF."""
    grid = grid_on_support(dist)
    # Compute true CDF on the grid points
    cdf = dist.cdf(grid)
    # Compute empirical CDF on the grid points
    samples = dist.sample([NUM_SAMPLES])
    assert (
        samples.shape == torch.Size([NUM_SAMPLES]) + dist.batch_shape + dist.event_shape
    )
    ecdf = (samples.unsqueeze(1) < grid).float().mean(0)
    return torch.allclose(cdf, ecdf, atol=tolerance)


@pytest.mark.parametrize("dist", EXAMPLES)
def test_sf(dist, tolerance=1e-4):
    """Test if cdf, sf, log_cdf, log_sf, log_prob and log_hazard are compatible."""
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


@pytest.mark.parametrize("dist", EXAMPLES)
def test_density(dist, tolerance=1e-5):
    """Compare two ways for computing the PDF of the distribution.

    1. Using the log_prob method.
    2. Derivative of the survival function: p(x) = -d/dx S(x).
    """
    x = grid_on_support(dist)
    x = x.requires_grad_()
    sf = dist.sf(x)
    sf.sum().backward()
    pdf1 = -x.grad
    pdf2 = dist.log_prob(x.detach()).exp()
    assert torch.allclose(pdf1, pdf2, atol=tolerance)


@pytest.mark.parametrize("dist", EXAMPLES)
def test_isf(dist, tolerance=1e-5, num_grid_points=100, delta=1e-3):
    """Check that the inverse survival function is implemented correctly."""
    x = torch.linspace(delta, 1 - delta, num_grid_points)
    # Change shape to (num_grid_points,) + dist.batch_shape for compatibility
    expanded_shape = x.shape + (1,) * len(dist.batch_shape)
    x = x.view(expanded_shape).expand(x.shape + dist.batch_shape)
    assert torch.allclose(x, dist.sf(dist.isf(x)), atol=tolerance)
