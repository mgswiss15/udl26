"""Mixture of Gaussians fitted by the EM algorithm - written for any number of variables d.

Shapes:  data X (N, d)   weights pis (K,)   means mus (K, d)   covariances Sigmas (K, d, d)
         responsibilities resp (N, K)
"""
import torch

from udl.distributions import REG, log_gaussian  # provided


def e_step(X, pis, mus, Sigmas):
    """Expectation step.

    Returns resp (N, K), the probability that example i comes from component k under the current parameters,
    and the log-likelihood=sum_i log p(x_i) of the data under the current parameters (a float).
    """
    # TODO: for every example and component the log of (weight x density of the component), normalise over components
    raise NotImplementedError
    return resp, log_likelihood


def m_step(X, resp):
    """Maximisation step. Returns the new pis (K,), mus (K, d), Sigmas (K, d, d).

    Add REG * identity to every new covariance so that it stays invertible.
    """
    N, d = X.shape
    K = resp.shape[1]
    # TODO: treat the responsibilities as fractional counts and compute weights, means and covariances of every component
    raise NotImplementedError
    return pis, mus, Sigmas


def init_gmm(X, k):
    """Starting point for EM: initial means, covariance for every component, and weights.
    
    Returns initial pis, mus, and Sigmas.
    """
    # TODO: k random data points as means, the data covariance for every component, equal weights.
    raise NotImplementedError
    return pis, mus, Sigmas


def gmm_em(X, k, n_iter=100):
    """Fit a mixture of k Gaussians to X (N, d).

    Returns pis, mus, Sigmas and the list of log-likelihoods (floats), one for every iteration.
    """
    pis, mus, Sigmas = init_gmm(X, k)  # provided starting point
    history = []
    for _ in range(n_iter):
        # TODO: one EM iteration - remember the log-likelihood before you update the parameters
        pass
    return pis, mus, Sigmas, history


def sample_gmm(pis, mus, Sigmas, n):
    """Generate n new examples (n, d) from the mixture."""
    # TODO: first choose the component of every sample, then sample from the Gaussian of that component
    raise NotImplementedError


def n_params(k, d):
    """Number of free parameters of a mixture of k Gaussians with full covariance matrices in d dimensions."""
    # TODO: count free numbers in the means, in the (symmetric) covariance matrices, and in the weights (which sum to 1)
    raise NotImplementedError
