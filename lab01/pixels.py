"""Independent-pixel model of binary images: every pixel has its own coin, unrelated to all other pixels."""
import torch


def fit_pixel_probs(B):
    """Learn the model. B: (N, D) binary images (D pixels, values 0 or 1). Returns p (D,): probability that each pixel is on."""
    # TODO: for every pixel the share of training images in which it is on
    raise NotImplementedError


def sample_pixels(p, n):
    """Generate n new binary images (n, D): every pixel is switched on with its own probability, independently."""
    # TODO: one separate coin flip per pixel and image
    raise NotImplementedError
