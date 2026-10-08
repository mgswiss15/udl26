"""Helpers used only in Lab 1 (week01): toy data, histogram and Bernoulli densities, plots. Provided - you do not implement these."""
import torch

from udl.plotting import BLUE, ORANGE, RED, to_numpy


# ---- data ----------------------------------------------------------------------------------

def make_waiting_times(n: int = 200, seed: int = 0) -> torch.Tensor:
    """Waiting times [min] of two groups of restaurant guests (quick and slow dishes), shape (n,)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g, dtype=torch.float64) < 0.5
    quick = 3.3 + 0.8 * torch.randn(n, generator=g, dtype=torch.float64)
    slow = 8.0 + 1.1 * torch.randn(n, generator=g, dtype=torch.float64)
    return torch.where(z, quick, slow)


def make_orders_2d(n: int = 200, seed: int = 0) -> torch.Tensor:
    """(waiting time [min], bill [EUR]) of two groups of restaurant guests, shape (n, 2).
    Inside each group a longer wait goes with a bigger bill."""
    g = torch.Generator().manual_seed(seed)
    f64 = torch.float64

    def group(mean, sd, rho):
        cov = torch.tensor([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]], dtype=f64)
        return torch.tensor(mean, dtype=f64) + torch.randn(n, 2, generator=g, dtype=f64) @ torch.linalg.cholesky(cov).T

    z = torch.rand(n, generator=g, dtype=f64) < 0.5
    return torch.where(z[:, None], group((3.3, 18.0), (0.8, 3.5), 0.6), group((8.0, 34.0), (1.1, 5.0), 0.6))


# ---- densities ----------------------------------------------------------------------------

def hist_logpdf(x, probs, edges):
    """Log-density of a 1D histogram model: log(bin probability / bin width).
    -inf for examples in an empty bin or outside the range.  x: (M,)  ->  (M,)."""
    n = len(probs)
    b = torch.bucketize(x, edges, right=True) - 1
    inside = (b >= 0) & (b < n)
    bc = b.clamp(0, n - 1)
    p = torch.where(inside, probs[bc], torch.zeros_like(x))
    width = torch.where(inside, (edges[1:] - edges[:-1])[bc], torch.ones_like(x))
    return torch.log(p / width)


def hist2d_logpdf(X, probs, edges_x, edges_y):
    """Log-density of a 2D histogram model: log(cell probability / cell area).
    -inf for examples in an empty cell or outside the range.  X: (M, 2)  ->  (M,)."""
    nx, ny = probs.shape
    i = torch.bucketize(X[:, 0], edges_x, right=True) - 1
    j = torch.bucketize(X[:, 1], edges_y, right=True) - 1
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    ic, jc = i.clamp(0, nx - 1), j.clamp(0, ny - 1)
    p = torch.where(inside, probs[ic, jc], torch.zeros_like(X[:, 0]))
    area = torch.where(inside, (edges_x[1:] - edges_x[:-1])[ic] * (edges_y[1:] - edges_y[:-1])[jc], torch.ones_like(X[:, 0]))
    return torch.log(p / area)


def bernoulli_logpdf(B, p, eps=1e-3):
    """Log-probability of binary images B (N, D) when pixel j is on with probability p[j] (independent pixels).
    p is clipped to [eps, 1 - eps] so that a pixel never seen on in training does not give -inf."""
    p = p.clamp(eps, 1 - eps)
    return (B * torch.log(p) + (1 - B) * torch.log(1 - p)).sum(1)


# ---- plots ---------------------------------------------------------------------------------

def plot_1d_fit(ax, data, samples, logpdf=None, title="", xlim=(0, 14), bins=28, edges=None):
    """Histogram of the data (blue), of the samples from the model (orange outline), model density (red, optional).
    logpdf: function mapping a tensor of shape (M,) to log densities of shape (M,)."""
    ax.hist(to_numpy(data), bins=bins, range=xlim, density=True, color=BLUE, edgecolor="#163c69", alpha=0.8, label="data")
    if torch.is_tensor(edges):
        bins_samples = len(edges - 1)
        xlim_samples = (edges[0].item(), edges[-1].item())
    else:
        bins_samples, xlim_samples = bins, xlim
    ax.hist(to_numpy(samples), bins=bins_samples, range=xlim_samples, density=True, histtype="step", color=ORANGE, lw=1.8, label="samples")
    if logpdf is not None:
        xs = torch.linspace(xlim[0], xlim[1], 400, dtype=torch.float64)
        ax.plot(to_numpy(xs), to_numpy(torch.exp(logpdf(xs))), color=RED, lw=2, label="model density")
    ax.set_xlim(*xlim)
    ax.set_xlabel("waiting time [min]")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_2d_fit(ax, data, samples, title="", xlim=(0, 12), ylim=(0, 60), labels=("waiting time [min]", "bill [EUR]")):
    """Scatter of data (blue) and of samples from the model (orange)."""
    data, samples = to_numpy(data), to_numpy(samples)
    ax.scatter(data[:, 0], data[:, 1], s=8, color="#729fcf", label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=8, color=ORANGE, alpha=0.6, label="samples")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(fontsize=8)
