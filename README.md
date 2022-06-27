# survival_distributions

This package extends the functionality of univariate distributions in [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html)
by implementing several new methods:
- `sf`: survival function (complementary CDF)
- `logsf`: logarithm of the survival function (negative cumulative hazard function)
- `logcdf`: logarithm of the CDF
- `log_hazard`: logarithm of the hazard function (logarithm of the failure rate)
- `isf`: inverse of the survival function
- `sample_cond`: instead of sampling from the full support of the distribution, 
generate samples between `lower_bound` and `upper_bound`
 
This is especially useful when working with
[temporal point processes](https://shchur.github.io/blog/2020/tpp1-conditional-intensity/)
or [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis).

Naive implementation based on existing PyTorch functionality (e.g., 
`torch.log(1.0 - dist.cdf(x))` for `logsf`) will often not be as accurate and numerically 
stable as the implementation provided by `survival_distributions`.
Hopefully, these methods will be implemented in PyTorch [sometime in the future](https://github.com/pytorch/pytorch/issues/52973), 
but this package provides an alternative for the time being.

See [`DISTRIBUTIONS.md`](https://github.com/shchur/survival_distributions/blob/main/DISTRIBUTIONS.md) for more details about the implemented functions and supported distributions.

## Installation
1. Install the latest version of [PyTorch](https://pytorch.org/get-started/locally/).
2. Install `survival_distributions`
```bash
pip install git+https://github.com/shchur/survival-distributions.git
```

## Supported distributions

### Numerically stable implementation
For these distributions we provide a numerically stable implementation of `logsf`.
- `Exponential`
- `Logistic`
- `LogLogistic`
- `MixtureSameFamily`
- `TransformedDistribution`
- `Uniform`
- `Weibull`

### Naive implementation 
For these distributions we implement `logsf(x)` as `log(1.0 - dist.cdf(x))`, which is less 
numerically stable.
- `LogNormal`
- `Normal`


## Usage
The package provides a drop-in replacement for `torch.distributions`, so you can just modify your code as follows.

**Old code**
```python
import torch

dist = torch.distributions.Exponential(rate=torch.tensor(2.0))
x = torch.tensor(1.5)

log_survival_proba = torch.log(1.0 - dist.cdf(x))
```
**New code**
```python
import torch
import survival_distributions as sd

dist = sd.Exponential(rate=torch.tensor(2.0))
x = torch.tensor(1.5)

log_survival_proba = dist.logsf(x)
```