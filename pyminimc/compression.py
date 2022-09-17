import numpy as np
import pandas as pd
from pyminimc import util
from collections import OrderedDict
from itertools import chain, product
from scipy.interpolate import RegularGridInterpolator
from typing import Literal, Sequence


def adaptive_coarsen(
    true_dfs: Sequence[pd.DataFrame],
    coarse_dfs: Sequence[pd.DataFrame],
    rel_frobenius_norm_tol: float = 1e-3,
    abs_linf_norm_tol: float = 1.0,
) -> Sequence[pd.DataFrame]:
    """
    Adaptively removes points from a list of DataFrames using a greedy
    algorithm that minimizes the global (frobenius) and local (linf) error
    introduced by linear interpolation. Stops when `rel_frobenius_norm_tol` or
    `abs_linf_norm_tol` is exceeded.

    Parameters
    ----------
    true_dfs
        Reference DataFrames to be used for error comparison
    coarse_df
        DataFrames to be used as starting point for grid coarsening
    rel_frobenius_norm_tol
        Maximum allowed frobenius norm of residuals across all DataFrames
        before coarsening stops. Norm is normalized by the frobenius norm of
        `true_dfs`.
    abs_linf_norm_tol
        Maximum allowed absolute l-infinity norm of residuals across all
        DataFrames before coarsening stops.

    Returns
    -------
    List of DataFrames with least important gridpoints removed
    """
    # linearly interpolate coarse_dfs so that axes match with true_dfs
    interp_dfs = [
        interpolate_df(coarse_df, true_df)
        for true_df, coarse_df in zip(true_dfs, coarse_dfs)
    ]
    # convert each DataFrame into Series
    (true_seriess, coarse_seriess, interp_seriess) = (
        [df.stack().stack() for df in dfs]
        for dfs in (true_dfs, coarse_dfs, interp_dfs)
    )
    # convert each Series into NDAarray and list of axis values
    (
        (true_arrays, true_axess),
        (coarse_arrays, coarse_axess),
        (interp_arrays, _),
    ) = (
        [
            [
                series.values.reshape(series.index.levshape)
                for series in seriess
            ],
            [
                [level.values for level in series.index.levels]
                for series in seriess
            ],
        ]
        for seriess in (true_seriess, coarse_seriess, interp_seriess)
    )
    # create mapping from coarse indices to corresponding true (fine) indices
    coarse_true_idx_map = [
        [
            np.array(np.searchsorted(true_axis, coarse_axis))
            for true_axis, coarse_axis in zip(true_axes, coarse_axes)
        ]
        for true_axes, coarse_axes in zip(true_axess, coarse_axess)
    ]
    # compute normalizing factor for l2 norm of true_array
    true_l2_norm = np.sqrt(
        sum(np.sum(np.square(true_array)) for true_array in true_arrays)
    )
    elapsed = 0
    while True:
        # after looping through each possible gridpoint, best_idx will either
        # be a tuple of (subset_idx, axis_idx, coarse_idx) or None
        best_idx = None
        best_rel_frobenius_norm = np.inf
        best_abs_linf_norm = np.inf
        best_interped = None
        best_interped_true_idxs = None

        subset_idxs = range(len(coarse_axess))
        residuals = [
            interp_arrays[idx] - true_arrays[idx] for idx in subset_idxs
        ]
        subset_sum_sqr_residuals = np.array(
            [np.sum(np.square(residuals[idx])) for idx in subset_idxs]
        )
        subset_max_abs_residuals = np.array(
            [np.max(np.abs(residuals[idx])) for idx in subset_idxs]
        )
        for subset_idx in subset_idxs:
            inactive_subset_idxs = np.array(
                [idx for idx in subset_idxs if idx != subset_idx]
            )
            subset_coarse_true_idx_map = coarse_true_idx_map[subset_idx]
            subset_residuals = residuals[subset_idx]
            true_axes = true_axess[subset_idx]
            subset_interp_array = interp_arrays[subset_idx]
            true_array = true_arrays[subset_idx]
            sum_subset_sum_sqr_residuals = np.sum(
                subset_sum_sqr_residuals[inactive_subset_idxs]
            )
            max_subset_max_abs_residuals = max(subset_max_abs_residuals)

            axis_idxs = range(len(coarse_axess[subset_idx]))
            for axis_idx in axis_idxs:
                inactive_axes_idxs = tuple(
                    (idx for idx in axis_idxs if idx != axis_idx)
                )
                axis_coarse_true_idx_map = subset_coarse_true_idx_map[axis_idx]
                axis_sum_sqr_residuals = np.sum(
                    np.square(subset_residuals), axis=inactive_axes_idxs
                )
                true_axis = true_axes[axis_idx]
                resize_shapes = [1 if a != axis_idx else -1 for a in axis_idxs]

                coarse_idxs = range(
                    1, len(coarse_axess[subset_idx][axis_idx]) - 1
                )
                for coarse_idx in coarse_idxs:
                    # get the true axis indices that will be used as the left/
                    # right boundaries for the interpolated points
                    left_true_idx = axis_coarse_true_idx_map[coarse_idx - 1]
                    right_true_idx = axis_coarse_true_idx_map[coarse_idx + 1]
                    # get fine axis indices for new points that have to be
                    # interpolated
                    interp_true_idxs = np.arange(
                        left_true_idx + 1, right_true_idx
                    )
                    # compute the interpolation factor for indices on the true
                    # axis that will be interpolated
                    interp_factor = (
                        (true_axis[interp_true_idxs] - true_axis[left_true_idx])
                        / (true_axis[right_true_idx] - true_axis[left_true_idx])
                    ).reshape(resize_shapes)
                    # get the previous iteration's interpolated values to use
                    # use as the left/right faces for the interpolated points
                    left_face = subset_interp_array.take(
                        [left_true_idx], axis=axis_idx
                    )
                    right_face = subset_interp_array.take(
                        [right_true_idx], axis=axis_idx
                    )
                    # interpolated points
                    interp_vals = left_face + interp_factor * (
                        right_face - left_face
                    )
                    # compute the residuals affected
                    true_vals = np.take(
                        true_array, interp_true_idxs, axis=axis_idx
                    )
                    affected_residuals = interp_vals - true_vals
                    # add up all contributions to l2 norm of residuals
                    total_sum_sqr_residual = np.sqrt(
                        sum_subset_sum_sqr_residuals
                        + np.sum(
                            np.delete(axis_sum_sqr_residuals, interp_true_idxs)
                        )
                        + np.sum(np.square(affected_residuals))
                    )
                    # add up all contributions to linf norm of residuals
                    total_max_abs_residual = max(
                        max_subset_max_abs_residuals,
                        np.max(np.abs(affected_residuals)),
                    )
                    # determine if current gridpoint is best
                    rel_frobenius_norm = total_sum_sqr_residual / true_l2_norm
                    abs_linf_norm = total_max_abs_residual
                    is_better = (
                        rel_frobenius_norm < best_rel_frobenius_norm
                        and rel_frobenius_norm < rel_frobenius_norm_tol
                        and abs_linf_norm < abs_linf_norm_tol
                    )
                    if is_better:
                        best_idx = (subset_idx, axis_idx, coarse_idx)
                        best_rel_frobenius_norm = rel_frobenius_norm
                        best_abs_linf_norm = abs_linf_norm
                        best_interped = interp_vals
                        best_interped_true_idxs = interp_true_idxs
        if best_idx is None:
            break
        subset_idx, axis_idx, coarse_idx = best_idx
        # update coarse axes/array
        keep_idx = np.delete(
            np.arange(coarse_axess[subset_idx][axis_idx].size), coarse_idx
        )
        coarse_axess[subset_idx][axis_idx] = coarse_axess[subset_idx][axis_idx][
            keep_idx
        ]
        coarse_arrays[subset_idx] = coarse_arrays[subset_idx].take(
            keep_idx, axis=axis_idx
        )
        # update interp axes/array
        # https://stackoverflow.com/a/42657219/5101335
        idx = [slice(None)] * interp_arrays[subset_idx].ndim
        idx[axis_idx] = best_interped_true_idxs
        interp_arrays[subset_idx][tuple(idx)] = best_interped
        # print diagnostics
        diagnostics = OrderedDict()
        diagnostics[
            "fine idx".rjust(10)
        ] = f"{coarse_true_idx_map[subset_idx][axis_idx][coarse_idx]:10}"
        diagnostics[
            "abs. l-inf norm residuals".rjust(27)
        ] = f"{best_abs_linf_norm:27.8E}"
        diagnostics[
            "rel. frobenius norm residuals".rjust(31)
        ] = f"{best_rel_frobenius_norm:31.8E}"
        diagnostics[
            "coarse axes shapes".rjust(70)
        ] = f"{[coarse_array.shape for coarse_array in coarse_arrays]}".rjust(
            70
        )
        if elapsed % 25 == 0:
            hlines = "".join(
                ("-" * len(s.strip())).rjust(len(s)) for s in diagnostics.keys()
            )
            headings = "".join(diagnostics.keys())
            print("\n".join((hlines, headings, hlines)))
        print("".join(diagnostics.values()))
        elapsed += 1
        # update mapping from coarse to true axes
        coarse_true_idx_map[subset_idx][axis_idx] = coarse_true_idx_map[
            subset_idx
        ][axis_idx][keep_idx]
    # return coarsened DataFrames
    coarse_dfs = [
        pd.DataFrame(
            pd.Series(
                coarse_array.flatten(),
                index=pd.MultiIndex.from_product(
                    coarse_axes, names=true_dfs[0].stack().stack().index.names
                ),
            )
            .unstack()
            .unstack()
        )
        for coarse_array, coarse_axes in zip(coarse_arrays, coarse_axess)
    ]
    return coarse_dfs


def truncate(df: pd.DataFrame, rank: int) -> pd.DataFrame:
    """
    Parameters
    ----------
    df
        DataFrame which will be approximated using SVD
    rank
        Rank of low-rank approximation

    Returns
    -------
    Low-rank approximation of DataFrame
    """
    U, S, Vt = np.linalg.svd(df, full_matrices=False)
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    Vt_truncated = Vt[:rank, :]
    return pd.DataFrame(
        U_truncated * S_truncated @ Vt_truncated,
        index=df.index,
        columns=df.columns,
    )


def remove_nonmonotonic_cdfs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df
        DataFrame which may contain nonmonotonic CDF

    Returns
    -------
    A DataFrame where each nonmonotonic CDF has been removed
    """
    # a monotonic entry is one which is greater than or equal to all entries
    # that appeared previously
    monotonic_nondecreasing = (df >= np.maximum.accumulate(df)).all(axis=1)
    # return monotonic CDF
    df = df.loc[monotonic_nondecreasing]
    # check monoticity
    assert (df.diff().iloc[1:] > 0).all().all()
    return df


def interpolate_df(
    coarse_df: pd.DataFrame,
    fine_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Linearly interpolate missing values from finer grid

    Parameters
    ----------
    coarse_df
        DataFrame whose missing values will be linearly interpolated
    fine_df
        DataFrame whose index and columns will be used to points to interpolate
        if they do not appear in `coarse_df`. It will also be used to obtain
        the value at max CDF if not present in `coarse_df`.
    """
    coarse_flattened = coarse_df.stack().stack()
    coarse_array = coarse_flattened.values.reshape(
        coarse_flattened.index.levshape
    )
    coarse_axes = [level.values for level in coarse_flattened.index.levels]
    fine_flattened = fine_df.stack().stack()
    fine_axes = [level.values for level in fine_flattened.index.levels]
    interped = RegularGridInterpolator(coarse_axes, coarse_array)(
        np.fromiter(chain(*product(*fine_axes)), float).reshape(
            fine_flattened.index.levshape + (-1,)
        )
    )
    return (
        pd.Series(interped.flatten(), index=fine_flattened.index)
        .unstack()
        .unstack()
    )


def split(
    full_df: pd.DataFrame, split_on: Literal["E", "beta"], splits: Sequence[int]
) -> Sequence[pd.DataFrame]:
    """
    Splits a DataFrame along columns

    Parameters
    ----------
    full_df
        The full DataFrame which will be split
    split_on
        The name of the MultiIndex level for which split indices will be given
    splits
        Indices which serve as the leftmost index of each partition. The first
        partition is implied to have index zero. Passing N splits will return
        N+1 partitions.
    """
    boundaries = np.concatenate(
        ([0.0], np.array(full_df.columns.unique(split_on)[splits]), [np.inf])
    )
    return [
        full_df.loc[
            :,
            (left_boundary <= full_df.columns.get_level_values(split_on))
            & (full_df.columns.get_level_values(split_on) < right_boundary),
        ]
        for left_boundary, right_boundary in zip(
            boundaries[:-1], boundaries[1:]
        )
    ]


def apply_approximations(
    true_df: pd.DataFrame,
    split_on: Literal["E", "beta"],
    splits: Sequence[int],
    ranks: Sequence[int],
) -> Sequence[pd.DataFrame]:
    """
    Applies a sequence of approximations to a CDF DataFrame

    Parameters
    ----------
    true_df
        DataFrame which will be approximated
    split_on
        Name of the column Index level where the splits will occur
    splits
        Indices of `split_on` where splits will be made
    ranks
        Rank of each subset when performing SVD
    """
    subsets = split(true_df, split_on=split_on, splits=splits)
    # obtain low-rank approximations of each subset
    print("\nobtaining low-rank approximations...")
    truncated_subsets = [
        truncate(subset, rank) for subset, rank in zip(subsets, ranks)
    ]
    util.print_errors(true_df, pd.concat(truncated_subsets, axis="columns"))
    # remove nonmonotonic CDF points from each subset
    print("\nremoving nonmonotonic CDFs...")
    monotonic_subsets = [
        remove_nonmonotonic_cdfs(subset) for subset in truncated_subsets
    ]
    util.print_errors(
        true_df,
        pd.concat(
            [
                interpolate_df(monotonic_subset, subset)
                for monotonic_subset, subset in zip(monotonic_subsets, subsets)
            ],
            axis="columns",
        ),
    )
    # adaptively coarsen each subset
    print("\nadaptively coarsening...")
    coarsened_subsets = adaptive_coarsen(subsets, monotonic_subsets)
    return coarsened_subsets
