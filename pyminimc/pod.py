from pyminimc import constants, util
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd


# bound cross section
sigma_free = 4.095600e1 / 2  # 2 hydrogens per H2O
sigma_bound = sigma_free * ((constants.A + 1) / constants.A) ** 2


def process_E_T(args):
    """
    Generates PDFs in beta, conditional PDFs in alpha given beta, and CDFs
    thereof

    Parameters
    ----------
    args : tuple
        (S(a,b,T) DataFrame, Incident energy in eV, Temperature in K)
    """
    sab_df, E, T = args
    T_betas = np.array(sorted(sab_df.loc[:, :, T].index.unique("beta")))
    # valid beta values
    min_beta = -E / (constants.k * T)
    max_beta = 20
    min_beta_index = np.searchsorted(
        T_betas, -min_beta
    )  # as range end, will never include min_beta
    max_beta_index = np.searchsorted(
        T_betas, max_beta
    )  # as range end, will never include max_beta
    betas = np.concatenate(
        (
            [min_beta],
            -np.flip(T_betas[1:min_beta_index]),
            T_betas[:max_beta_index],
            [max_beta],
        )
    )
    alpha_cdfs = {}
    alpha_pdfs = {}
    beta_pdf = pd.Series(0, index=betas)  # pdf is zero at min_beta & max_beta
    for beta in betas[1:-1]:
        # energy-independent alpha distribution
        S_values = sab_df.xs(
            (beta if beta > 0 else -beta, T), level=("beta", "T")
        )["S"]
        # energy-dependent alpha distribution, valid alpha values
        min_alpha = np.square(
            np.sqrt(E) - np.sqrt(E + beta * constants.k * T)
        ) / (constants.A * constants.k * T)
        max_alpha = np.square(
            np.sqrt(E) + np.sqrt(E + beta * constants.k * T)
        ) / (constants.A * constants.k * T)
        S_at_min_alpha, S_at_max_alpha = np.interp(
            [min_alpha, max_alpha], S_values.index, S_values
        )
        min_alpha_index = np.searchsorted(
            S_values.index, min_alpha, side="right"
        )  # will never include min_alpha
        max_alpha_index = np.searchsorted(
            S_values.index, max_alpha
        )  # as range end, will never include max_alpha
        S_values = pd.concat(
            (
                pd.Series({min_alpha: S_at_min_alpha}),
                S_values.iloc[min_alpha_index:max_alpha_index],
                pd.Series({max_alpha: S_at_max_alpha}),
            )
        )
        alpha_cdf = util.lin_log_cum_trapz(S_values)
        alpha_integral = alpha_cdf.iloc[-1]
        alpha_cdfs[beta] = alpha_cdf / alpha_integral
        alpha_pdfs[beta] = (S_values / alpha_integral).rename(beta)
        beta_pdf.loc[beta] = np.exp(-beta / 2.0) * alpha_integral
    # convert pdf to cdf
    beta_cdf = util.lin_log_cum_trapz(beta_pdf)
    beta_integral = beta_cdf.iloc[-1]
    beta_cdf /= beta_integral
    beta_pdf /= beta_integral
    total_inelastic_xs = (
        sigma_bound * constants.A * constants.k * T / (4 * E) * beta_integral
    )
    return beta_pdf, alpha_pdfs, beta_cdf, alpha_cdfs, total_inelastic_xs


def process_b_T(sab_s, max_alpha):
    """
    Generates conditional CDF in alpha given beta and temperature. Returns None
    if there is zero /probability of the given beta being sampled.

    Parameters
    ----------
    sab_s : pd.Series
        A pd.Series containing a MultiIndex. The MultiIndex levels are
        temperature `T`, `beta`, and `alpha`. There is only a single value of
        `T` and `beta` while multiple `alpha` values must be present. The
        values are corresponding value of S(a,b,T).
    max_alpha : double
        largest alpha in the dataset
    """
    sab_s.index = sab_s.index.droplevel(["T", "beta"])
    # set endpoints to zero
    sab_s.loc[0] = 0
    sab_s.loc[max_alpha] = 0
    sab_s = sab_s.sort_index()
    E_independent_alpha_cdf = util.lin_log_cum_trapz(sab_s)
    E_independent_alpha_integral = E_independent_alpha_cdf.iloc[-1]
    # if integral is zero, return nothing
    if E_independent_alpha_integral == 0:
        E_independent_alpha_cdf = None
    else:
        E_independent_alpha_cdf /= E_independent_alpha_integral
    return E_independent_alpha_cdf


def beta_functional_expansion(
    sab_df, E_min=1e-5, E_max=4.0, n_Es=1000, n_cdfs=1000, order=None
):
    """
    Computes the CDF in beta at various incident energies and temperatures,
    then performs a functional expansion in temperature at various incident
    energies and CDF values.

    Parameters
    ----------
    sab_df : pd.DataFrame
        S(a,b,T) DataFrame
    E_min : float, optional
        Minimum incident energy in eV. Will be included in final energy grid.
    E_max : float, optional
        Maximum incident energy in eV. Will not be included in final energy
        grid.
    n_Es : int, optional
        Approximate number of incident energies (equally spaced in lethargy)
    n_cdfs : int, optional
        Number of CDF values to use
    order : int, optional
        Expansion order for proper orthogonal decomposition. Setting to None
        will return the full expansion.

    Returns
    -------
    pd.DataFrame
        Expansion coefficients in temperature for beta given a given set of
        incident energies and CDF values.

    Todo
    ----
    Use DataFrames for intermediate steps instead of numpy arrays. This is how
    alpha_functional_expansion() does it.
    """
    # Populate `beta_cdfs`, a 2D array where the first index corresponds to
    # incident energy and the second index corresponds to temperature. Each
    # element of this array is a cumulative distribution function for beta.
    df_Ts = np.array(sorted(sab_df.index.unique("T")))
    # equally spaced-lethargy intervals, do not include zero lethargy
    assert E_min > 0
    lethargies = np.linspace(0, np.log(E_max / E_min), num=n_Es + 1)[:0:-1]
    Es = E_max * np.exp(-lethargies)
    with Pool(processes=8) as pool:
        # incident energy, temperature pairs
        E_T_values = np.array([(sab_df, E, T) for E in Es for T in df_Ts])
        results = np.array(
            [
                [x[2], x[4]]
                for x in tqdm(
                    pool.imap(func=process_E_T, iterable=E_T_values),
                    total=len(E_T_values),
                )
            ],
            dtype=object,
        ).reshape(len(Es), len(df_Ts), -1)
        beta_cdfs = results[:, :, 0]
        inelastic_xs = pd.DataFrame(results[:, :, 1], index=Es, columns=df_Ts)
    F = np.linspace(0, 1, n_cdfs)
    beta_df = pd.DataFrame(
        np.nan,
        index=pd.Index(F, name="CDF"),
        columns=pd.MultiIndex.from_product((Es, df_Ts), names=("E", "T")),
    )
    # a good beta grid for the alpha functional expansion
    unique_betas = np.unique(np.abs(beta_df))
    N_betas = 1000
    selected_betas = unique_betas[
        np.round(np.linspace(0, unique_betas.size - 1, N_betas)).astype(int)
    ]
    # interpolate to fill in selected CDF values
    for E, x_E in zip(Es, beta_cdfs):
        for T, beta_cdf in zip(df_Ts, x_E):
            beta_df.loc[:, (E, T)] = np.interp(F, beta_cdf, beta_cdf.index)
    # perform proper orthogonal decomposition
    U_df, S_df, V_df = util.to_svd_dfs(beta_df, order)
    order = S_df.size if order is None else order
    # set energy units to MeV
    V_df.index = V_df.index.set_levels(
        V_df.index.unique(level="E") * 1e-6, level="E"
    )
    # reconstruct
    beta_df_reconstructed = util.reconstruct_from_svd_dfs(U_df, S_df, V_df)
    # check that CDFS are monotonic for certain T values
    print(
        f"RMSE: {np.sqrt(((beta_df_reconstructed - beta_df)**2).mean().mean())}"
    )
    is_monotonic = beta_df_reconstructed.apply(lambda s: s.is_monotonic)
    print(
        f"Of {Es.size} incident energies and {df_Ts.size} target "
        f"temperatures, {is_monotonic.sum()} of {is_monotonic.size} "
        f"({is_monotonic.sum() / is_monotonic.size * 100}%) have "
        f"monotonic beta as a function of CDF"
    )
    if not is_monotonic.all():
        print("The following CDFs are not monotonic:")
        print(beta_df_reconstructed.loc[:, ~is_monotonic])
    return U_df, S_df, V_df


def alpha_functional_expansion(sab_df, selected_betas, n_cdfs=1000, order=None):
    """
    Computes the energy-independent conditional CDF in alpha given beta at
    various beta values and temperatures, then performs a functional expansion
    in CDF at various beta and temperature values.

    Parameters
    ----------
    sab_df : pd.DataFrame
        S(a,b,T) DataFrame
    selected_betas : np.ndarray
        Beta values to use
    n_cdfs : int, optional
        Number of CDF values to use
    order : int, optional
        Expansion order for proper orthogonal decomposition. Setting to None
        will return the full expansion.
    """
    df_Ts = np.array(sorted(sab_df.index.unique("T")))
    selected_betas = set(selected_betas)

    def common_beta_grid(group):
        """
        Modifies beta values at this temperature to conform with common beta
        grid.
        """
        nonlocal sab_df
        group.index = group.index.droplevel("T")
        # add betas from selected_betas which are not already in group
        new_betas = selected_betas.difference(group.index.unique("beta"))
        new_df = pd.DataFrame(
            np.nan,
            columns=group.index.unique("alpha"),
            index=pd.Index(new_betas, name="beta"),
        )
        combined_df = pd.concat(
            [group.unstack("alpha")["S"], new_df]
        ).sort_index()
        # S(a,b) above maximum beta at this temperature is zero
        combined_df.loc[
            combined_df.index > group.index.unique("beta").max()
        ] = 0
        # interpolate linearly in beta and linearly in ln(S) (interpolation
        # scheme 4 in ENDF)
        return (
            np.exp(
                np.log(combined_df).interpolate(
                    method="index", axis="index", limit_area="inside"
                )
            )
            .loc[list(selected_betas)]
            .stack()
        )

    common_beta_sab_df = sab_df.groupby("T").apply(common_beta_grid)
    # alpha grids do not have to match across T-beta pairs, but they do have to
    # have the same minimum and maximum values
    print("computing alpha CDFs...")
    largest_alpha = sab_df.index.unique("alpha").max()
    print(f"largest alpha: {largest_alpha}")
    # choose number of CDF points we want to use; don't include 0 or 1
    F = np.linspace(0, 1, n_cdfs + 2)[1:-1]
    # Some T-beta values will have all-zeros and result in missing entries for
    # such T-beta values.
    alpha_df = (
        common_beta_sab_df.groupby(["T", "beta"])
        .apply(process_b_T, largest_alpha)
        .groupby(["T", "beta"])
        .apply(
            lambda s: pd.Series(
                np.interp(F, s, s.index.get_level_values("alpha")),
                index=pd.Index(F, name="CDF"),
            )
        )
    )
    # construct matrix to decompose with POD
    alpha_df_pod_form = alpha_df.unstack(["beta", "T"])
    # add back T-beta pairs which didn't have a well-defined alpha CDF
    undefined_cdfs = common_beta_sab_df.unstack("alpha").index.difference(
        alpha_df.unstack("CDF").index
    )
    alpha_df_pod_form[undefined_cdfs] = np.nan
    alpha_df_pod_form = alpha_df_pod_form.sort_index(axis="columns")

    # impute NaN entries: https://stackoverflow.com/a/35611142/5101335
    nan_entries = alpha_df_pod_form.isna()
    alpha_df_pod_form = alpha_df_pod_form.T.fillna(
        value=alpha_df_pod_form.T.mean()
    ).T
    # repeat multiple times until convergence
    converged = False
    prev_trace_norm = np.nan
    while not converged:
        U, S, Vt = np.linalg.svd(alpha_df_pod_form, full_matrices=False)
        order = S.size if order is None else order
        A = U[:, :order] @ np.diag(S[:order]) @ Vt[:order, :]
        alpha_df_pod_form = alpha_df_pod_form.where(~nan_entries, A)
        trace_norm = S.sum()
        abs_rel_diff = np.abs((trace_norm - prev_trace_norm) / prev_trace_norm)
        print(f"trace norm abs rel diff: {abs_rel_diff}", end=" " * 10 + "\r")
        if abs_rel_diff < 1e-8:
            break
        else:
            prev_trace_norm = trace_norm
    U_df = pd.DataFrame(
        {"coefficient": U[:, :order].flatten()},
        index=pd.MultiIndex.from_product(
            [F, range(order)], names=["CDF", "order"]
        ),
    )
    S_df = pd.DataFrame(
        {"coefficient": S[:order]},
        index=pd.MultiIndex.from_product([range(order)], names=["order"]),
    )
    V_df = pd.DataFrame(
        {"coefficient": Vt.T[:, :order].flatten()},
        index=pd.MultiIndex.from_product(
            [alpha_df.index.unique("beta"), df_Ts, range(order)],
            names=["beta", "T", "order"],
        ),
    )
    # reconstruct
    alpha_df_reconstructed = pd.DataFrame(
        U[:, :order] @ np.diag(S[:order]) @ Vt[:order, :],
        index=alpha_df_pod_form.index,
        columns=alpha_df_pod_form.columns,
    )
    print(
        f"RMSE: {np.sqrt(((alpha_df_reconstructed - alpha_df_pod_form)**2).mean().mean())}"
    )
    # check that CDFS are monotonic for certain T values
    is_monotonic = alpha_df_reconstructed.apply(lambda s: s.is_monotonic)
    print(
        f"{is_monotonic.sum()} of {is_monotonic.size} "
        f"({is_monotonic.sum() / is_monotonic.size * 100}%) have "
        f"monotonic alpha as a function of CDF"
    )
    if not is_monotonic.all():
        print("The following CDFs are not monotonic:")
        print(alpha_df_reconstructed.loc[:, ~is_monotonic])
    return U_df, S_df, V_df
