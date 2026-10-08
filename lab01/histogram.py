"""Histogram models of one and two variables: learn from examples, generate new examples.

All data are float tensors. Bins and cells are described by their edges.
"""
import torch


def fit_histogram(x, edges):
    """Learn a 1D histogram model.

    x:     (N,) training examples
    edges: (B + 1,) increasing bin edges
    Returns probs: (B,) probability of every bin; the probabilities sum to 1.
    """
    # TODO: probability of a bin = share of the training examples that fall into it
    raise NotImplementedError


def sample_histogram(probs, edges, n):
    """Generate n new examples (n,) from the 1D histogram model. Inside a bin the model is uniform."""
    # TODO: choose a bin for every sample according to the bin probabilities, then a point inside that bin
    raise NotImplementedError


def fit_histogram_2d(X, edges_x, edges_y):
    """Learn a 2D histogram model.

    X:       (N, 2) training examples
    edges_x: (Bx + 1,) bin edges of the first variable;  edges_y: (By + 1,) of the second
    Returns probs: (Bx, By) probability of every cell; the probabilities sum to 1.
    """
    # TODO: probability of a cell = share of the training examples that fall into it
    raise NotImplementedError


def sample_histogram_2d(probs, edges_x, edges_y, n):
    """Generate n new examples (n, 2) from the 2D histogram model. Inside a cell the model is uniform."""
    # TODO: choose a cell for every sample according to the cell probabilities, then a point inside that cell
    raise NotImplementedError
