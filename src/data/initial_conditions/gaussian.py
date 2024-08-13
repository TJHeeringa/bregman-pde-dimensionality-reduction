from dataclasses import dataclass

import torch


@dataclass
class GaussianParamSet:
    """Parameters for the Gaussian class.

    :param mu: center of the gaussian
    :param sigma: variance of the gaussian
    """

    mu: torch.Tensor
    sigma: torch.Tensor


class Gaussian:
    def __init__(self, param_set: GaussianParamSet):
        self.mu = param_set.mu
        self.sigma = param_set.sigma
        self.dim = param_set.sigma.dim()
        if self.dim == 1:
            self.inv_sigma = 1 / self.sigma
            self.det = self.sigma
        else:
            self.inv_sigma = torch.inverse(self.sigma)
            self.det = torch.linalg.det(self.sigma)

    def parameter_set(self):
        return GaussianParamSet(mu=self.mu, sigma=self.sigma)

    @staticmethod
    def parameter_set_class():
        return GaussianParamSet

    def __call__(self, x):
        """
        Input x has size (batch_size, dim).
        """
        delta = x - self.mu

        if self.dim == 1:
            mahalanobis_sqrd = delta**2 * self.inv_sigma
        else:
            mahalanobis_sqrd = torch.dot(delta, torch.matmul(delta, self.inv_sigma))

        pi_constant = (2 * torch.tensor(torch.pi)) ** self.dim
        return 1 / torch.sqrt(pi_constant * self.det) * torch.exp(-0.5 * mahalanobis_sqrd)
