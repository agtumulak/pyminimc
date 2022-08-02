"""
Classes for representing estimator results
"""
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict


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
        # number of boundaries matches number of bins
        assert (
            np.prod([b.size - 1 for b in self._boundaries.values()])
            == counts.size
        )

    def assert_axis_exists(self, axis_name: str):
        assert axis_name in self._boundaries, (
            f"Axis name '{axis_name}' "
            f"not found in [{', '.join(self._boundaries.keys())}]."
        )

    def marginalize(self, marginalized_name: str):
        self.assert_axis_exists(marginalized_name)
        counts = self._counts.reshape(
            [b.size - 1 for b in self._boundaries.values()]
        )
        # get index that will be marginalized
        marginalized_index = [
            i
            for i, k in enumerate(self._boundaries.keys())
            if k == marginalized_name
        ][0]
        # sum counts along marginalized axis
        self._counts = np.sum(counts, axis=marginalized_index)
        # drop marginalized boundaries
        del self._boundaries[marginalized_name]

    def plot(self, dependent_name: str, **kwargs):
        self.assert_axis_exists(dependent_name)
        marginalized_names = set(self._boundaries) - set([dependent_name])
        for marginalized_name in marginalized_names:
            self.marginalize(marginalized_name)
        # bind to shorter names
        counts = self._counts
        count_errs = np.sqrt(counts * (1 - counts) / self._N)
        boundaries = self._boundaries[dependent_name]
        # plot as probability density function
        widths = np.diff(boundaries)
        midpoints = boundaries[:-1] + 0.5 * widths
        densities = counts / widths
        density_errs = count_errs / widths
        x_vals = np.concatenate(([boundaries[0]], midpoints, [boundaries[-1]]))
        y_vals = np.concatenate((np.array([0]), densities, np.array([0])))
        y_err = np.concatenate((np.array([0]), density_errs, np.array([0])))
        y_lower = np.subtract(y_vals, y_err)
        y_upper = np.add(y_vals, y_err)
        plt.plot(x_vals, y_vals, **kwargs)
        plt.fill_between(x_vals, y_lower, y_upper, alpha=0.2)


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
        super().plot(dependent_name, label="mcnp")


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
        super().plot(dependent_name, label="minimc")
