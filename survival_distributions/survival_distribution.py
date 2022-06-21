import torch


class SurvivalDistribution(torch.distributions.Distribution):
    """Base class for all survival distributions.

    By default, all distributions are expected to implement logsf, log_prob and isf.
    Other methods (logcdf, log_hazard, rsample, rsample_conditional) are computed based
    on them.
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

    def rsample_conditional(
        self, sample_shape=torch.Size(), lower_bound=None, upper_bound=None
    ):
        shape = self._extended_shape(sample_shape)
        # TODO: Should I detach u_min and u_max?
        if lower_bound is not None:
            u_min = self.cdf(lower_bound)
        else:
            u_min = 0.0
        if upper_bound is not None:
            u_max = self.cdf(upper_bound)
        else:
            u_max = 1.0
        u = (u_max - u_min) * self._new_tensor(shape).uniform_() + u_min
        return self.isf(u)

    def sample_conditional(
        self, sample_shape=torch.Size(), lower_bound=None, upper_bound=None
    ):
        with torch.no_grad():
            return self.sample_conditional(
                sample_shape=sample_shape,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
