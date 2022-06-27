Table of contents
- [Background](#background)
  - [Functions describing a probability distribution](#functions-describing-a-probability-distribution)
  - [Sampling](#sampling)
- [List of probability distributions](#list-of-probability-distributions)
  - [Exponential](#exponential)
  - [Logistic](#logistic)
  - [LogLogistic](#loglogistic)
  - [LogNormal](#lognormal)
  - [Normal](#normal)
  - [Uniform](#uniform)
  - [Weibull](#weibull)


## Background
### Functions describing a probability distribution
There exist multiple ways to describe a probability distribution of a univariate random variable $X$.
In machine learning, we usually work with the following two functions:
1. [Probability density function (PDF)](https://en.wikipedia.org/wiki/Probability_density_function)
$$p(x) = \Pr(X \in [x, x + dt))$$
2. [Cumulative distribution function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
$$
\begin{align*}
F(x) &= \Pr(X \le x)\\
&= \int_{-\infty}^{x} p(u) du
\end{align*}
$$
However, there also exist other options that can be convenient in practice.
For example, when working with temporal point processes or survival analysis, we often prefer the following two functions:

3. [Survival function (SF)](https://en.wikipedia.org/wiki/Survival_function)
$$
\begin{align*}
S(x) &= \Pr(X \ge x)\\
&= 1 - F(x)
\end{align*}
$$
4. [Hazard function (a.k.a., failure rate or intensity)](https://en.wikipedia.org/wiki/Failure_rate)
$$
\begin{align*}
h(x) &= \Pr(X \in [x, x + dt) \mid X \ge x)\\
&= \frac{p(x)}{S(x)}
\end{align*}
$$

Each of these four functions uniquely defines the distribution of $X$.
Therefore, if we specify either $p(x)$, $F(x)$, $S(x)$ or $h(x)$, the other 3 functions are immediately defined as well.

### Sampling
The **inverse survival function** $S^{-1}$ provides us with a simple way to generate samples of $X$ using $\operatorname{Uniform}([0, 1])$ noise:
$$
\begin{align*}
u &\sim \operatorname{Uniform}([0, 1])\\
x &= S^{-1}(u)
\end{align*}
$$

The above procedure generates a sample from the entire support of the distribution.
However, we can easily adapt it to only draw samples from an interval $[x_{\text{min}}, x_{\text{max}}] \subseteq \mathbb{R}$.
$$
\begin{align*}
a &= S(x_{\text{max}})\\
b &= S(x_{\text{min}})\\
u &\sim \operatorname{Uniform}([a, b])\\
x &= S^{-1}(u)
\end{align*}
$$
Here is an example where this can be useful.
Suppose $X$ corresponds to inter-arrival time between events (e.g., failures of some machine in a factory).
If we know that no failure occurred in the last $x_{\text{min}} = 50$ days, we can use the above procedure to draw samples conditioned on this fact.


## List of probability distributions
### Exponential
- Parameters
    - rate $\lambda > 0$
- Support: $(0, \infty)$
- PDF
$$p(x) = \lambda \exp(- \lambda x)$$
- SF
$$S(x) = \exp(-\lambda x)$$
- CDF
$$F(x) = 1 - \exp(-\lambda x)$$
- Inverse SF
$$S^{-1}(u) = -\frac{1}{\lambda} \log (u)$$


### Logistic
- Parameters
    - location $\mu$
    - scale $s > 0$
- Support: $\mathbb{R}$
- PDF
$$p(x) = \frac{\exp\left(\frac{x - \mu}{s}\right)}{s \cdot \left(1 + \exp\left(\frac{x - \mu}{s}\right)\right)^2}$$
- SF
$$S(x) = \frac{1}{1 + \exp\left(\frac{x - \mu}{s}\right)}$$
- CDF
$$F(x) = \frac{1}{1 + \exp\left(-\frac{x - \mu}{s}\right)}$$
- Inverse SF
$$S^{-1}(u) = s \cdot \log\left(\frac{1-u}{u}\right) + \mu$$


### LogLogistic
- Parameters
    - location $\mu$
    - scale $s > 0$
- Support: $(0, \infty)$
- PDF
$$p(x) = \frac{\exp\left(\frac{\log(x) - \mu}{s}\right)}{x \cdot s \cdot \left(1 + \exp\left(\frac{\log(x) - \mu}{s}\right)\right)^2}$$
- SF
$$S(x) = \frac{1}{1 + \exp\left(\frac{\log(x) - \mu}{s}\right)}$$
- CDF
$$F(x) = \frac{1}{1 + \exp\left(-\frac{\log(x) - \mu}{s}\right)}$$
- Inverse SF
$$S^{-1}(u) = \exp\left(s \cdot \log\left(\frac{1-u}{u}\right) + \mu\right)$$



### LogNormal
- Parameters
  - location $\mu$
  - scale $s > 0$
- Support $(0, \infty)$
- PDF
$$p(x) = \frac{1}{x s\sqrt{2\pi}}\exp\left(-\frac{(\log (x) - \mu)^2}{2s^2}\right)$$
- CDF
$$F(x) = \Phi\left(\frac{\log(x)-\mu}{s}\right)$$
  where $\Phi$ is the CDF of the standard normal distribution.
- SF
$$S(x) = 1 - \Phi\left(\frac{\log(x)-\mu}{s}\right)$$
- Inverse SF
$$S^{-1}(u) = \exp\left(s \cdot \Phi^{-1}(u) + \mu\right)$$


### Normal
- Parameters
  - location $\mu$
  - scale $s > 0$
- Support $\mathbb{R}$
- PDF
$$p(x) = \frac{1}{s\sqrt{2\pi}}\exp\left(-\frac{(x - \mu)^2}{2s^2}\right)$$
- CDF
$$F(x) = \Phi\left(\frac{x-\mu}{s}\right)$$
  where $\Phi$ is the CDF of the standard normal distribution.
- SF
$$S(x) = 1 - \Phi\left(\frac{x-\mu}{s}\right)$$
- Inverse SF
$$S^{-1}(u) = s \cdot \Phi^{-1}(u) + \mu$$

### Uniform
- Parameters
    - lower boundary $a$
    - upper boundary $b > a$
- Support: $(a, b)$
- PDF
$$p(x) = \frac{1}{b - a}$$
- CDF
$$F(x) = \frac{x - a}{b - a}$$
- SF
$$F(x) = \frac{b - x}{b - a}$$
- Inverse SF
$$S^{-1}(u) = b - u \cdot (b - a)$$

### Weibull
- Parameters
    - rate $b > 0$
    - concentration $k > 0$
- Support: $(0, \infty)$
- PDF
$$p(x) = b k x^{k-1} \exp(-bx^k)$$
- SF
$$S(x) = \exp(-bx^k)$$
- CDF
$$F(x) = 1 - \exp(-bx^k)$$
- Inverse SF
$$S^{-1}(u) = \left(-\frac{1}{b} \log (u)\right)^{\frac{1}{k}}$$