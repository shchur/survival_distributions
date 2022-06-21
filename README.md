# survival_distributions

This package extends the functionality of univariate distributions in [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html)
by implementing several new methods:
- `sf`: survival function (complementary CDF)
- `logsf`: logarithm of the survival function (negative cumulative hazard function)
- `log_hazard`: logarithm of the hazard function (logarithm of the failure rate)
- `isf`: inverse of the survival function
- `sample_cond`: instead of sampling from the full support of the distribution, 
generate samples between `lower_bound` and `upper_bound`
 
This is especially useful when working with
[temporal point processes](https://shchur.github.io/blog/2020/tpp1-conditional-intensity/)
or [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis).

Naive implementation based on existing PyTorch functionality (e.g., 
`torch.log(1 - p.cdf(x))` for `logsf`) will often not be as accurate and numerically 
stable as the implementation provided by `survival_distributions`.
Hopefully, these methods will be implemented in PyTorch [sometime in the future](https://github.com/pytorch/pytorch/issues/52973), 
but this package provides an alternative for the time being.


### Supported distributions

| Name                      | `logsf`  | `log_hazard` | `isf`   |
|---------------------------|----------|--------------|---------|
| `Exponential`             | &check;  | &check;      | &check; |
| `Weibull`                 | &check;  | &check;      | &check; |

Legend:
- &check; — numerically stable implementation  available
- ? — naive (numerically unstable) implementation available 