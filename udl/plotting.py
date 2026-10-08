"""Plotting helpers shared by several labs."""
import numpy as np
import torch


def plot_data(ax, x, title: str = "", lim: float = 3.5) -> None:
    ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_density(logp_fn, ax, data=None, lim: float = 3.5, n: int = 200, title: str = "") -> None:
    """Plot exp(logp_fn(x)) on a grid. logp_fn maps a (M, 2) tensor to (M,) log-densities."""
    xs = torch.linspace(-lim, lim, n)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        p = logp_fn(grid).exp().reshape(n, n)
    ax.contourf(gx.numpy(), gy.numpy(), p.numpy(), levels=30, cmap="viridis")
    if data is not None:
        ax.scatter(data[:300, 0], data[:300, 1], s=2, c="white", alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)


BLUE, ORANGE, RED = "#a6bddb", "#ff6600", "#b31b1b"


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def plot_images(axes, images, title=""):
    """Show 28 x 28 images (flattened or not) on a row of axes: white on black."""
    axes = np.ravel(axes)
    for ax, im in zip(axes, images):
        ax.imshow(to_numpy(im).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        axes[0].set_title(title, fontsize=9, loc="left")
