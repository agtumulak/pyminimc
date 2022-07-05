"""
Functions for construcitng probability density functions from various sources
"""

from pyminimc import constants, pod
import numpy as np
import pandas as pd


def get_pdf_direct(sab_df, E, T, label=None):
    """
    Creates bivariate PDF in alpha and beta from given S(a,b,T) DataFrame

    Alpha grid is unionized across all possible beta values. Linear
    interpolation is used in alpha for missing values. Zero values are used
    outside valid alpha ranges.

    Parameters
    ----------
    sab_df : pd.DataFrame
        S(a,b,T) DataFrame
    E : float
        Incident energy in eV
    T : float
        Temperature in K
    """
    label = label if label else "direct"
    beta_pdf, alpha_pdfs, _, _, _ = pod.process_E_T((sab_df, E, T))
    for beta, p_beta in beta_pdf.iloc[1:-1].iteritems():
        alpha_pdfs[beta] *= p_beta
    return (
        pd.concat(alpha_pdfs.values(), names=alpha_pdfs.keys(), axis="columns")
        .interpolate(method="index", axis="index", limit_area="inside")
        .fillna(0)
        .stack()
        .rename_axis(["alpha", "beta"])
        .rename(label)
    )


def get_pdf_pod(
    beta_T_path,
    beta_S_path,
    beta_E_CDF_path,
    alpha_T_path,
    alpha_S_path,
    alpha_beta_CDF_path,
    E,
    T,
    max_alpha,
    label=None,
):
    """
    Reconstructs bivariate PDF in alpha and beta from proper orthogonal
    decomposition coefficients for alpha and beta

    Parameters
    ---------
    beta_T_path : string
        Path to HDF5 file containing temperature dependent coefficients for beta
    beta_S_path : string
        Path to HDF5 file containing singular values for beta
    beta_E_CDF : string
        Path to HDF5 file containing energy and CDF dependent coefficients for
        beta
    alpha_T_path : string
        Path to HDF5 file containing temperature dependent coefficients for alpha
    alpha_S_path : string
        Path to HDF5 file containing singular values for alpha
    alpha_E_CDF : string
        Path to HDF5 file containing beta and CDF dependent coefficients for
        alpha
    E : float
        Incident energy in eV
    T : float
        Temperature in K
    max_alpha : float
        Largest alpha in the dataset
    """
    label = label if label else "direct (pod reconstructed)"
    # Load data
    beta_T = pd.read_hdf(beta_T_path)
    beta_S = pd.read_hdf(beta_S_path)
    beta_E_CDF = pd.read_hdf(beta_E_CDF_path)
    alpha_T = pd.read_hdf(alpha_T_path)
    alpha_S = pd.read_hdf(alpha_S_path)
    alpha_beta_CDF = pd.read_hdf(alpha_beta_CDF_path)
    # evaluate beta at nearest E and nearest T
    Ts = beta_T.index.unique("T")
    nearest_T = Ts[np.argmin(np.abs(Ts - T))]
    Es = beta_E_CDF.index.unique("E")
    nearest_E = Es[np.argmin(np.abs(Es - E * 1e-6))]
    # Reconstruct beta CDF
    beta_cdf = pd.Series(
        (
            beta_T.loc[nearest_T].T.values
            @ np.diag(beta_S.values.flatten())
            @ beta_E_CDF.loc[nearest_E].unstack().T.values
        ).flatten(),
        index=beta_E_CDF.index.unique("CDF"),
    )
    # compute PDF
    beta_pdf = pd.Series(
        (beta_cdf.index[1:] - beta_cdf.index[:-1])
        / (beta_cdf.values[1:] - beta_cdf.values[:-1]),
        index=beta_cdf.values[:-1],
    )
    # evaluate alpha at (possibly different) nearest T and nearest beta
    Ts = alpha_T.index.unique("T")
    nearest_T = Ts[np.argmin(np.abs(Ts - T))]
    # Reconstruct alpha CDF
    alpha_cdf = pd.Series(
        (
            alpha_T.loc[nearest_T].T.values
            @ np.diag(alpha_S.values.flatten())
            @ alpha_beta_CDF.unstack().T.values
        ).flatten(),
        index=alpha_beta_CDF.unstack().index,
    ).unstack("beta")
    # append alpha CDFs for negative beta values
    alpha_betas = alpha_cdf.columns
    # find largest beta in alpha_betas which is strictly less than E / (constants.k * T)
    # we assume beta = 0 exists so result of searchsorted is >= 1
    min_beta = alpha_betas[np.searchsorted(alpha_betas, -beta_cdf.iloc[0]) - 1]
    neg_b_alpha_cdf = (
        alpha_cdf.loc[:, alpha_betas[1] : min_beta]  # don't include beta = 0
        .rename(columns=lambda x: -x)  # make beta labels negative
        .sort_index(axis="columns")
    )
    # find largest beta in alpha_betas which is strictly less than 20
    max_beta = alpha_betas[np.searchsorted(alpha_betas, 20) - 1]
    alpha_cdf = pd.concat(
        (neg_b_alpha_cdf, alpha_cdf.loc[:max_beta]), axis="columns"
    )
    # add endpoints
    alpha_cdf.loc[0, :] = 0
    alpha_cdf.loc[1, :] = max_alpha
    alpha_cdf = alpha_cdf.sort_index()
    # choose common alpha_grid
    alphas = np.linspace(0, max_alpha, 10000)

    # choose subset of alpha values
    def get_joint_probability(s):
        """
        Multiplies conditional probability in alpha with probability in beta
        """
        nonlocal alphas
        nonlocal E
        beta = s.name
        # skip values of beta which won't be reached
        if E + beta * constants.k * T < 0:
            return pd.Series(0, index=alphas)
        # choose correct subset of CDF values within min_alpha and max_alpha
        s = pd.Series(s.index, index=pd.Index(s, name="alpha"))
        # insert values for min_alpha and max_apha
        min_alpha = np.square(
            np.sqrt(E) - np.sqrt(E + beta * constants.k * T)
        ) / (constants.A * constants.k * T)
        max_alpha = np.square(
            np.sqrt(E) + np.sqrt(E + beta * constants.k * T)
        ) / (constants.A * constants.k * T)
        s.loc[min_alpha] = np.nan  # interpolate value later
        s.loc[max_alpha] = np.nan  # interpolate value later

        s = s.sort_index().interpolate(method="index").loc[min_alpha:max_alpha]
        # rescale CDF to be 0 at min_alpha and 1 at max_alpha
        s = (s - s.min()) / (s.max() - s.min())
        alpha_pdf = pd.Series(
            (s.values[1:] - s.values[:-1]) / (s.index[1:] - s.index[:-1]),
            index=s.index[:-1],
        )
        alpha_pdf.loc[min_alpha] = 0
        alpha_pdf.loc[max_alpha] = 0
        alpha_pdf = alpha_pdf.sort_index()
        # use common alpha grid
        new_alphas = set(alphas).difference(set(alpha_pdf.index))
        alpha_pdf = (
            pd.concat([alpha_pdf, pd.Series(np.nan, index=new_alphas)])
            .sort_index()
            .interpolate(method="index", limit_area="inside")
            .fillna(0)
            .loc[alphas]
        )
        # interpolate value of beta pdf
        nonlocal beta_pdf
        beta_pdf_value = np.interp(
            beta, beta_pdf.index, beta_pdf, left=0, right=0
        )
        return alpha_pdf * beta_pdf_value

    bivariate_pdf = alpha_cdf.apply(get_joint_probability).stack()
    bivariate_pdf.index.names = ["alpha", "beta"]
    return bivariate_pdf.rename(label)


def get_pdf_runsab(counts_path, *bounds_paths):
    """
    Creates bivariate PDF in alpha and beta from a list of counts and N paths
    to axis boundaries

    Parameters
    ----------
    counts_path : string
        Path to file containing counts separated by newlines. Counts are
        ordered with the last element in `bounds_paths` changing the fastest,
        and the first element in `bounds_paths` changing slowest.
    bounds_paths : sequence of strings
        paths to files containing bin boundaries.

    Returns
    -------
    pd.Series with a MultiIndex for each axis
    """
    with open(counts_path) as f:
        counts = np.array([int(l.strip()) for l in f.readlines()])
        counts = counts / counts.sum()
    widths, bin_edges = [], []
    for bounds_path in bounds_paths:
        with open(bounds_path) as f:
            bounds = np.array([float(x) for x in f.readlines()])
            widths.append(bounds[1:] - bounds[:-1])
            bin_edges.append(bounds[:-1])
    bin_areas = np.einsum("i,j->ij", *widths).reshape(-1)
    density = counts / bin_areas
    return pd.Series(
        density,
        index=pd.MultiIndex.from_product(bin_edges, names=["alpha", "beta"]),
        name="minimc",
    )


def get_pdf_mcnp(mctal_path, E, T, label=None):
    """
    Creates bivariate PDF in alpha and beta from an MCNP mctal file

    Parameters
    ----------
    mctal_path : string
        Path to mctal file
    E : float
        Incident energy in eV
    T : float
        Temperature in K
    """
    label = label if label else "mcnp"
    cosine_bounds = [
        -1.0,
    ]
    energy_bounds = [
        0.0,
    ]
    counts = []
    with open(mctal_path) as f:
        line = ""
        # skip to cosine bins
        while not line.startswith("c"):
            line = f.readline()
        line = f.readline()
        # parse cosine boundaries until energy boundaries section is reached
        while not line.startswith("e"):
            cosine_bounds.extend(float(x) for x in line.split())
            line = f.readline()
        line = f.readline()
        # parse energy boundaries until time boundaries section is reached
        while not line.startswith("t"):
            energy_bounds.extend(float(x) * 1e6 for x in line.split())
            line = f.readline()
        # skip to values
        while not line.startswith("vals"):
            line = f.readline()
        line = f.readline()
        # parse values until the tfc section is reached
        while not line.startswith("tfc"):
            counts.extend(float(x) for x in line.split()[::2])
            line = f.readline()
    # compute densities in cosine, eV space
    cosine_bounds = np.array(cosine_bounds)
    energy_bounds = np.array(energy_bounds)
    counts = np.array(counts)
    cosine_widths = cosine_bounds[1:] - cosine_bounds[:-1]
    energy_widths = energy_bounds[1:] - energy_bounds[:-1]
    bin_areas = np.einsum("i,j->ij", cosine_widths, energy_widths).reshape(-1)
    density = counts / bin_areas
    s = pd.Series(
        density,
        index=pd.MultiIndex.from_product(
            [cosine_bounds[:-1], energy_bounds[:-1]], names=["mu", "E"]
        ),
        name="mcnp",
    )

    def to_alpha_beta(s):
        beta = s.name
        out_E = E + beta * constants.k * T
        mus = s.index.get_level_values("mu")
        alphas = (E + out_E - 2 * mus * np.sqrt(E * out_E)) / (
            constants.A * constants.k * T
        )
        return pd.Series(s.values, index=alphas).rename_axis(
            index={"mu": "alpha"}
        )

    # multiply by jacobian
    s = (
        s
        * 0.5
        * constants.A
        * (constants.k * T) ** 2
        / (np.sqrt(s.index.get_level_values("E") * E))
    )
    s = (
        s.rename(index=lambda out_E: (out_E - E) / (constants.k * T), level="E")
        .rename_axis(index={"E": "beta"})
        .groupby("beta")
        .apply(to_alpha_beta)
    )
    return (
        s[~s.index.duplicated()]
        .unstack("beta")
        .interpolate(method="index", axis="index", limit_area="inside")
        .fillna(0)
        .stack()
        .rename_axis(["alpha", "beta"])
        .rename(label)
    )


def get_pdf_minimc(minimc_path, E, T, label=None):
    """
    Creates bivariate PDF in alpha and beta from minimc output
    """
    label = label if label else "minimc"
    with open(minimc_path) as f:
        line = ""
        # skip to cosine bins
        while not line.startswith("cosine"):
            line = f.readline()
        line = f.readline()
        cosine_bounds = [float(x) for x in f.readline().split(",")[:-1]]
        # skip to energy bins
        while not line.startswith("energy"):
            line = f.readline()
        line = f.readline()
        energy_bounds = [float(x) * 1e6 for x in f.readline().split(",")[:-1]]
        # skip to values
        while not line.startswith("mean"):
            line = f.readline()
        line = f.readline()
        counts = [float(x) for x in f.readline().split(",")[:-1]]
    # compute densities in cosine, eV space
    cosine_bounds = np.concatenate(([-np.inf], cosine_bounds, [np.inf]))
    energy_bounds = np.concatenate(([-np.inf], energy_bounds, [np.inf]))
    counts = np.array(counts)
    cosine_widths = cosine_bounds[1:] - cosine_bounds[:-1]
    energy_widths = energy_bounds[1:] - energy_bounds[:-1]
    bin_areas = np.einsum("i,j->ij", cosine_widths, energy_widths).reshape(-1)
    # compute densities and remove infinitely sized bins
    density = (
        (counts / bin_areas)
        .reshape(cosine_widths.size, energy_widths.size)[1:-1, 1:-1]
        .reshape(-1)
    )
    # create pd.Series
    cosine_midpoints = (cosine_bounds[2:-1] + cosine_bounds[1:-2]) / 2
    energy_midpoints = (energy_bounds[2:-1] + energy_bounds[1:-2]) / 2
    s = pd.Series(
        density,
        index=pd.MultiIndex.from_product(
            [cosine_midpoints, energy_midpoints], names=["mu", "E"]
        ),
        name="minimc",
    )

    # convert to alpha, beta space; #TODO: this is identical to mcnp version
    def to_alpha_beta(s):
        beta = s.name
        out_E = E + beta * constants.k * T
        mus = s.index.get_level_values("mu")
        alphas = (E + out_E - 2 * mus * np.sqrt(E * out_E)) / (
            constants.A * constants.k * T
        )
        return pd.Series(s.values, index=alphas).rename_axis(
            index={"mu": "alpha"}
        )

    # multiply by jacobian
    s = (
        s
        * 0.5
        * constants.A
        * (constants.k * T) ** 2
        / (np.sqrt(s.index.get_level_values("E") * E))
    )
    s = (
        s.rename(index=lambda out_E: (out_E - E) / (constants.k * T), level="E")
        .rename_axis(index={"E": "beta"})
        .groupby("beta")
        .apply(to_alpha_beta)
    )
    return (
        s[~s.index.duplicated()]
        .unstack("beta")
        .interpolate(method="index", axis="index", limit_area="inside")
        .fillna(0)
        .stack()
        .rename_axis(["alpha", "beta"])
        .rename(label)
    )
