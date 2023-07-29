"""
Classes for representing estimator results
"""
import numpy as np
from scipy.integrate import quadrature
from matplotlib import pyplot as plt
from collections import OrderedDict


class MarginalizedNonmultiplyingBinnedLeakageEstimator:
    """
    Like NonmultiplyingBinnedLeakageEstimator, but there is only one axis.
    """

    def __init__(self, N: int, counts: np.ndarray, boundaries: np.ndarray):
        self._N = N
        self._counts = counts
        self._boundaries = boundaries

    def as_midpoint_density_pairs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_errs = np.sqrt(self._counts * (1 - self._counts) / self._N)
        widths = np.diff(self._boundaries)
        midpoints = self._boundaries[:-1] + 0.5 * widths
        densities = self._counts / widths
        density_errs = count_errs / widths
        return midpoints, densities, density_errs

    def plot(self, **kwargs):
        midpoints, densities, density_errs = self.as_midpoint_density_pairs()
        x_vals = np.concatenate(
            ([self._boundaries[0]], midpoints, [self._boundaries[-1]])
        )
        y_vals = np.concatenate((np.array([0]), densities, np.array([0])))
        y_err = np.concatenate((np.array([0]), density_errs, np.array([0])))
        y_lower = np.subtract(y_vals, y_err * 3)
        y_upper = np.add(y_vals, y_err * 3)
        plt.plot(x_vals, y_vals, **kwargs)
        plt.fill_between(x_vals, y_lower, y_upper, linewidth=0, alpha=0.2, color='k')

    def l2_error(self, other):
        # bind to shorter names
        _, self_densities, _ = self.as_midpoint_density_pairs()
        _, other_densities, _ = other.as_midpoint_density_pairs()

        def func(x):
            # TODO: Figure out if x ever includes boundaries
            return np.square(
                self_densities[np.searchsorted(self._boundaries, x) - 1]
                - other_densities[np.searchsorted(other._boundaries, x) - 1]
            )

        return quadrature(func, self._boundaries[0], self._boundaries[-1])


class NonmultiplyingBinnedLeakageEstimator:
    """
    Models the result of a current estimator set up to measure leakage from a
    nonmultiplying medium.

    Each bin (cosine, energy, ...) in the estimator is modeled as a Bernoulli
    random variable. As a result, the variance of the mean reported by a
    transport code is not used since it can be expressed in terms of the mean
    and sample size. A source particle can only score in one bin since all the
    bins are mutually exclusive and leaves the system once it scores. The sum
    over all bins is less than or equal to one since not all source particles
    necessarily leak through the estimator.


    Parameters
    ----------
    N:
        Total number of samples used to produce estimates. Used to compute
        variance of the mean.
    counts:
        Flattened array of counts. Its size must be reshapeable to
    boundaries:
        One or more name/array pairs. Each array is a list of boundaries for a
        bin dimension (cosine, energy, etc...). The order each array appears
        corresponds to row-major appearance in ``counts``.
    """

    def __init__(
        self,
        N: int,
        counts: np.ndarray,
        boundaries: OrderedDict[str, np.ndarray],
    ):
        self._N = N
        self._counts = counts
        self._boundaries = boundaries
        # number of counts matches number of bins
        assert (
            np.prod([b.size - 1 for b in self._boundaries.values()])
            == counts.size
        )

    def marginalize(self, unmarginalized_name: str):
        # assert axis exists
        assert unmarginalized_name in self._boundaries, (
            f"Axis name '{unmarginalized_name}' "
            f"not found in [{', '.join(self._boundaries.keys())}]."
        )
        # compute argument for reshape
        shape = [b.size - 1 for b in self._boundaries.values()]
        # compute argument for sum
        sum_axes = tuple(
            [
                i
                for i, name in enumerate(self._boundaries)
                if name != unmarginalized_name
            ]
        )
        # return a copy
        return MarginalizedNonmultiplyingBinnedLeakageEstimator(
            self._N,
            np.sum(self._counts.reshape(shape), axis=sum_axes),
            self._boundaries[unmarginalized_name],
        )


class MCNP(NonmultiplyingBinnedLeakageEstimator):
    """
    Parses MCNP mctal files
    """

    def __init__(self, mctal_path: str):
        cosine_boundaries = [
            -1.0,
        ]
        energy_boundaries = [
            0.0,
        ]
        counts = []
        with open(mctal_path) as f:
            # read first line and get number of histories
            line = f.readline()
            N = int(line.split()[5])
            # skip to cosine bins
            while not line.startswith("c"):
                line = f.readline()
            line = f.readline()
            # parse cosine boundaries until energy boundaries section is reached
            while not line.startswith("e"):
                cosine_boundaries.extend(float(x) for x in line.split())
                line = f.readline()
            line = f.readline()
            # parse energy boundaries until time boundaries section is reached
            while not line.startswith("t"):
                energy_boundaries.extend(float(x) * 1e6 for x in line.split())
                line = f.readline()
            # skip to values
            while not line.startswith("vals"):
                line = f.readline()
            line = f.readline()
            # parse values until the tfc section is reached
            while not line.startswith("tfc"):
                counts.extend(float(x) for x in line.split()[::2])
                line = f.readline()
        super().__init__(
            N,
            np.array(counts),
            OrderedDict(
                [
                    ("cosine", np.array(cosine_boundaries)),
                    ("energy", np.array(energy_boundaries)),
                ]
            ),
        )

    def plot(self, dependent_name: str):
        super().marginalize(dependent_name).plot(label="mcnp")


class MiniMC(NonmultiplyingBinnedLeakageEstimator):
    """
    Parses MiniMC files
    """

    def __init__(self, minimc_path: str):
        with open(minimc_path) as f:
            line = f.readline()
            N = int(line.split()[0])
            # skip to cosine bins
            while not line.startswith("cosine"):
                line = f.readline()
            line = f.readline()
            cosine_bounds = np.concatenate(
                (
                    [-np.inf],
                    [float(x) for x in f.readline().split(",")[:-1]],
                    [np.inf],
                )
            )
            # skip to energy bins
            while not line.startswith("energy"):
                line = f.readline()
            line = f.readline()
            energy_bounds = np.concatenate(
                (
                    [-np.inf],
                    [float(x) * 1e6 for x in f.readline().split(",")[:-1]],
                    [np.inf],
                )
            )
            # skip to values
            while not line.startswith("mean"):
                line = f.readline()
            line = f.readline()
            counts = np.array([float(x) for x in f.readline().split(",")[:-1]])
        # remove infinite-width bins
        counts = counts.reshape(cosine_bounds.size - 1, energy_bounds.size - 1)[
            1:-1, 1:-1
        ]
        cosine_bounds = cosine_bounds[1:-1]
        energy_bounds = energy_bounds[1:-1]
        super().__init__(
            N,
            counts,
            OrderedDict(
                [
                    ("cosine", cosine_bounds),
                    ("energy", energy_bounds),
                ]
            ),
        )

    def plot(self, dependent_name: str):
        super().marginalize(dependent_name).plot(label="minimc")
