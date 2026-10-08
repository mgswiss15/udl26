"""Log-densities and starting points for Gaussian and mixture models (torch). Provided - you do not implement these."""
import math

import torch

REG = 1e-6  # added to covariance diagonals so they stay invertible


def log_gaussian(X, mu, Sigma):
    """Log-density of N(mu, Sigma) at the rows of X.  X: (N, d), mu: (d,), Sigma: (d, d)  ->  (N,)."""
    X = torch.atleast_2d(X)
    d = X.shape[1]
    L = torch.linalg.cholesky(Sigma)
    z = torch.linalg.solve_triangular(L, (X - mu).T, upper=False)  # (d, N)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    return -0.5 * (d * math.log(2 * math.pi) + logdet + (z**2).sum(0))


def gmm_logpdf(X, pis, mus, Sigmas):
    """Log-density of a Gaussian mixture at the rows of X.  X: (N, d)  ->  (N,)."""
    X = torch.atleast_2d(X)
    logp = torch.stack([torch.log(pis[k]) + log_gaussian(X, mus[k], Sigmas[k]) for k in range(len(pis))], dim=1)
    return torch.logsumexp(logp, dim=1)



