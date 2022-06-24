from typing import Optional

import torch
from torch.distributions import Categorical

from .survival_distribution import SurvivalDistribution


class MixtureSameFamily(torch.distributions.MixtureSameFamily, SurvivalDistribution):
    def __init__(
        self, mixture_distribution, component_distribution, validate_args=None
    ):
        super(MixtureSameFamily, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=validate_args,
        )

    def logsf(self, value):
        value = self._pad(value)
        log_sf_x = self.component_distribution.logsf(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)

    def sf(self, value):
        value = self._pad(value)
        sf_x = self.component_distribution.sf(value)
        mix_prob = self.mixture_distribution.probs
        return torch.sum(sf_x * mix_prob, dim=-1)

    def sample_cond(
        self,
        sample_shape: torch.Size = torch.Size(),
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # Since we know that the sample x > lower_bound, we have to adjust the
            # mixing probabilities as p(z_i = k) * Pr(x >= lower_bound | z_i = k)
            # mixture samples [n, B]
            if lower_bound is not None:
                above_lb_prob = self.component_distribution.sf(lower_bound)
            else:
                above_lb_prob = 1.0

            if upper_bound is not None:
                above_ub_prob = self.component_distribution.sf(upper_bound)
            else:
                above_ub_prob = 0.0

            conditional_mix_probs = self.mixture_distribution.probs * (
                above_lb_prob - above_ub_prob
            )
            mix_sample = Categorical(probs=conditional_mix_probs).sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample_cond(
                sample_shape=sample_shape,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)
