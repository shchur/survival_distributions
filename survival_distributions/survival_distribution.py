from typing import Optional

import torch


class SurvivalDistribution(torch.distributions.Distribution):
    """Base class for all survival distributions.

    By default, all distributions are expected to implement logsf, log_prob and isf.
    Other methods (logcdf, log_hazard, rsample, rsample_cond) are derived from them.
    """

    def logsf(self, value):
        """Logarithm of the survival function.

        This is equal to the negative cumulative (integrated) hazard function.
        """
        raise NotImplementedError

    def sf(self, value):
        """Survival function.

        Also known as complementary cumulative distribution function (complementary CDF).

        References:
            https://en.wikipedia.org/wiki/Survival_function
        """
        return self.logsf(value).exp()

    def logcdf(self, value):
        """Logarithm of the cumulative distribution function."""
        return torch.log1p(-self.sf(value))

    def cdf(self, value):
        """Cumulative distribution function evaluated at value."""
        return 1.0 - self.sf(value)

    def isf(self, u):
        """Inverse of the survival function."""
        raise NotImplementedError

    def log_hazard(self, value):
        """Logarithm of the hazard function evaluated at value.

        Also known as failure rate.

        Hazard function is defined as PDF divided by the survival function.

        References:
            https://en.wikipedia.org/wiki/Survival_analysis#Hazard_function_and_cumulative_hazard_function
        """
        return self.log_prob(value) - self.logsf(value)

    def _new_tensor(self, shape):
        """Create new tensor with same dtype and device as distribution parameters."""
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self._new_tensor(shape).uniform_()
        return self.isf(u)

    def rsample_cond(
        self,
        sample_shape: torch.Size = torch.Size(),
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        """Draw a sample that is guaranteed to be between lower_bound and upper_bound.

        We generate a sample on the interval [upper_bound, upper_bound] as
            a = sf(upper_bound)
            b = sf(lower_bound)
            u ~ Uniform([a, b])  # note the switched order of a, b!
            x = isf(u)

        If neither lower_bound or upper_bound are given, this is equivalent to the
        standard rsample
            u ~ Uniform([0, 1])
            x = isf(u)
        """
        shape = self._extended_shape(sample_shape)
        # Sample u_full from Uniform([0, 1])
        u_full = self._new_tensor(shape).uniform_()
        if lower_bound is not None:
            lb = torch.as_tensor(lower_bound, dtype=u_full.dtype, device=u_full.device)
            u_max = self.sf(lb)
        else:
            u_max = 1.0
        if upper_bound is not None:
            ub = torch.as_tensor(upper_bound, dtype=u_full.dtype, device=u_full.device)
            u_min = self.sf(ub)
        else:
            u_min = 0.0
        # Equivalent to sampling u ~ Uniform([u_min, u_max])
        u = (u_max - u_min) * u_full + u_min
        # TODO: Should we detach u here?
        return self.isf(u)

    def sample_cond(
        self, sample_shape=torch.Size(), lower_bound=None, upper_bound=None
    ):
        with torch.no_grad():
            return self.rsample_cond(
                sample_shape=sample_shape,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
