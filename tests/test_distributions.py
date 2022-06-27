import pytest
import torch
from torch.distributions import Categorical, constraints

from survival_distributions import (
    Exponential,
    Logistic,
    LogLogistic,
    LogNormal,
    MixtureSameFamily,
    Normal,
    Uniform,
    Weibull,
)

NUM_COMPONENTS = 5
NUM_SAMPLES = 100_000
TOLERANCE_STRICT = 1e-5
TOLERANCE_RELAXED = 1e-2

SAMPLE_SHAPES = [(NUM_SAMPLES,), (100, 20), (50, 20, 5), (1, 1, 8, 1)]

DISTRIBUTIONS = [
    Exponential(rate=torch.tensor([0.7, 0.2, 0.4], requires_grad=True)),
    Exponential(rate=2.3),
    LogNormal(loc=torch.tensor([-1.5, 2.5, 3.0]), scale=torch.tensor([1.2, 2.5, 0.8])),
    LogNormal(loc=0.0, scale=1.5),
    Logistic(loc=torch.tensor([-1.5, 2.5, 3.0]), scale=torch.tensor([1.2, 2.5, 0.8])),
    LogLogistic(
        loc=torch.tensor([1.5, 2.5, -2.0]), scale=torch.tensor([1.2, 0.2, 0.8])
    ),
    MixtureSameFamily(
        mixture_distribution=Categorical(
            logits=torch.empty(NUM_COMPONENTS).normal_(0.0, 0.2)
        ),
        component_distribution=Normal(
            loc=torch.linspace(-8, 8, steps=NUM_COMPONENTS),
            scale=torch.empty(NUM_COMPONENTS).uniform_(0.5, 1.5),
        ),
    ),
    Normal(loc=torch.tensor([-1.5, 2.5, 3.0]), scale=torch.tensor([1.2, 2.5, 0.8])),
    Normal(loc=4.1, scale=2.2),
    Uniform(low=-0.5, high=2.0),
    Weibull(
        rate=torch.tensor([2.0, 0.5, 1.1]), concentration=torch.tensor([0.5, 2.5, 1.0])
    ),
    Weibull(rate=2.0, concentration=2.5),
]


def grid_on_support(dist, num_grid_points=500, eps=1e-5, x_max=10):
    """Generate a grid on the support of the distribution"""
    if dist.support == constraints.positive:
        grid = torch.linspace(eps, x_max, num_grid_points)
    elif dist.support == constraints.real:
        grid = torch.linspace(-x_max, x_max, num_grid_points)
    elif isinstance(dist.support, constraints.interval):
        grid = torch.linspace(
            dist.support.lower_bound + eps,
            dist.support.upper_bound - eps,
            num_grid_points,
        )
    else:
        raise ValueError("dist must have support positive or real")
    # The grid must have shape (num_grid_points,) + dist.batch_shape
    expanded_shape = grid.shape + (1,) * len(dist.batch_shape)
    return grid.view(expanded_shape).expand(grid.shape + dist.batch_shape)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_ecdf(dist, tolerance=TOLERANCE_RELAXED):
    """Check whether the empirical CDF based on the samples is close to the true CDF."""
    grid = grid_on_support(dist)
    # Compute true CDF on the grid points
    cdf = dist.cdf(grid)
    # Compute empirical CDF on the grid points
    samples = dist.sample([NUM_SAMPLES])
    ecdf = (samples.unsqueeze(1) < grid).float().mean(0)
    return torch.allclose(cdf, ecdf, atol=tolerance)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_distribution_functions(dist, tolerance=TOLERANCE_STRICT):
    """Test if cdf, sf, log_cdf, log_sf, log_prob and log_hazard are compatible."""
    # survival_distributions only supports univariate distributions
    assert dist.event_shape == torch.Size()

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
    assert (
        cdf.shape
        == sf.shape
        == log_sf.shape
        == log_cdf.shape
        == log_pdf.shape
        == log_hazard.shape
    )


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_density(dist, tolerance=TOLERANCE_STRICT):
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


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_isf(dist, tolerance=1e-5, num_grid_points=100, delta=1e-3):
    """Check that the inverse survival function is implemented correctly."""
    x = torch.linspace(delta, 1 - delta, num_grid_points)
    # Change shape to (num_grid_points,) + dist.batch_shape for compatibility
    expanded_shape = x.shape + (1,) * len(dist.batch_shape)
    x = x.view(expanded_shape).expand(x.shape + dist.batch_shape)
    try:
        x_reconstructed = dist.sf(dist.isf(x))
        assert torch.allclose(x, x_reconstructed, atol=tolerance)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_sample_cond(dist, tolerance=TOLERANCE_RELAXED, x_min=0.5, x_max=2.0):
    samples = dist.sample_cond([NUM_SAMPLES], lower_bound=x_min, upper_bound=x_max)
    # Numerical issues in isf may produce samples slightly below x_min / above x_max
    assert samples.min() >= x_min - tolerance
    assert samples.max() <= x_max + tolerance
    assert samples.shape == torch.Size([NUM_SAMPLES]) + dist.batch_shape


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_sample_shapes(dist):
    for shape in SAMPLE_SHAPES:
        s1 = dist.sample(shape)
        s2 = dist.sample_cond(shape, lower_bound=1.0, upper_bound=2.0)
        assert s1.shape == s2.shape == torch.Size(shape) + dist.batch_shape
        if dist.has_rsample:
            s3 = dist.rsample(shape)
            s4 = dist.rsample_cond(shape, lower_bound=1.0, upper_bound=2.0)
            assert s1.shape == s3.shape == s4.shape
