"""Single Gaussian models of any number of variables: learn from examples, generate new examples."""
import torch


def fit_gaussian(X):
    """Learn a Gaussian model.

    X: (N, d) training examples (one variable: d = 1)
    Returns mean (d,) and covariance (d, d). Divide by N, not N - 1.
    """
    # TODO: mean vector and covariance matrix of the examples
    raise NotImplementedError


def sample_gaussian(mean, cov, n):
    """Generate n new examples (n, d) from the Gaussian with the given mean (d,) and covariance (d, d)."""
    # TODO: start from independent standard normal noise and transform it to the required mean and covariance
    raise NotImplementedError


def remove_correlations(cov):
    """Covariance (d, d) of the model in which every variable is independent of the others (same variances)."""
    # TODO: keep the variances, replace all covariances between different variables by zero
    raise NotImplementedError
