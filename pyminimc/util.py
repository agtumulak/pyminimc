import numpy as np
import pandas as pd
import re

def parse_file7(mf7_path: str):
    """
    Parse ENDF File 7 file and return pandas DataFrame.

    Parameters
    ----------
    mf7_path
        Path to File 7
    """

    def to_float(endf_float_string):
        """Convert ENDF-style float string to float."""
        pattern = re.compile(r"\d([+-])")
        return float(pattern.sub(r"E\1", endf_float_string))

    with open(mf7_path, mode="r") as sab_file:
        # skip headers
        while True:
            if sab_file.readline().endswith("1 7  4\n"):
                break
        # skip to relevant part
        for _ in range(4):
            next(sab_file)
        N_beta = int(sab_file.readline().split()[0])
        alphas, betas, Ts, Ss = [], [], [], []
        for _ in range(N_beta):
            # Read first temperature, beta block
            entries = sab_file.readline().split()
            temp, beta = (to_float(x) for x in entries[:2])
            N_temp = int(entries[2]) + 1
            # The first temperature is handled differently: alpha and
            # S(alpha, beta) values are given together.
            N_alpha = int(sab_file.readline().split()[0])
            N_full_rows, remainder = divmod(N_alpha, 3)
            for _ in range(N_full_rows + (remainder != 0)):
                # Everything after column 66 is ignored
                line = sab_file.readline()
                doubles = [
                    to_float(line[start : start + 11])
                    for start in range(0, 66, 11)
                    if not line[start : start + 11].isspace()
                ]
                for alpha, S in zip(doubles[::2], doubles[1::2]):
                    alphas.append(alpha)
                    betas.append(beta)
                    Ts.append(temp)
                    Ss.append(S)
            # The remaining temperatures are handled uniformly
            N_full_rows, remainder = divmod(N_alpha, 6)
            for _ in range(N_temp - 1):
                temp, beta = (
                    to_float(x) for x in sab_file.readline().split()[:2]
                )
                # Subsequent betas use the first beta's alpha grid.
                unique_alphas = (a for a in alphas[:N_alpha])
                for _ in range(N_full_rows + (remainder != 0)):
                    line = sab_file.readline()[:66]
                    for S in [
                        to_float(line[start : start + 11])
                        for start in range(0, 66, 11)
                        if not line[start : start + 11].isspace()
                    ]:
                        alphas.append(next(unique_alphas))
                        betas.append(beta)
                        Ts.append(temp)
                        Ss.append(S)
        df = pd.DataFrame.from_dict(
            {"alpha": alphas, "beta": betas, "T": Ts, "S": Ss}
        )
        # rescale alpha and beta due to LAT flag in LEAPR
        df["alpha"] = df["alpha"] * 293.6 / df["T"]
        df["beta"] = df["beta"] * 293.6 / df["T"]
        df = df.set_index(["beta", "alpha", "T"])
        return df


def marginalize(s, axis="beta"):
    """
    Converts a bivariate PDF into a univariate PDF in given axis

    Parameters
    ----------
    s : pd.Series
        Bivariate PDF in alpha and beta to plot. MultiIndex must be beta
        followed by alpha.
    axis : {'alpha', 'beta'}, optional
        The axis of the resulting univariate PDF

    Todo
    ----
    Fix issue where values probability mass near edges is dropped/ignored
    """
    return s.groupby(axis).apply(
        lambda s: np.trapz(s.values, s.index.droplevel(axis))
    )


def lin_log_cum_trapz(s):
    """
    Integrates a Series with trapezoid rule using linear in index and
    logarithmic in value interpolation

    Parameters
    ----------
    s : pd.Series
       Series to integrate
    """
    x, y = s.index, s.values
    numerators = y[1:] - y[:-1]
    denominators = np.log(y[1:]) - np.log(y[:-1])
    # https://stackoverflow.com/a/37977222/5101335
    ratios = np.divide(
        numerators, denominators, out=numerators, where=denominators != 0
    )
    ratios = np.nan_to_num(ratios)
    return pd.Series(
        np.concatenate(([0], np.cumsum(ratios * (x[1:] - x[:-1])))),
        index=s.index,
    )


def to_svd_dfs(
    df: pd.DataFrame, order: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform Singular Value Decomposition on DataFrame and return factorization
    as separate DataFrames

    Parameters
    ----------
    df
        Original DataFrame to decompose using singular value dcomposition
    order
        Order to truncate singular value decomposition. Setting to None will
        set order to rank of DataFrame.
    """
    U, S, Vt = np.linalg.svd(df, full_matrices=False)
    order = S.size if order is None else order
    U_df = pd.DataFrame(
        {"coefficient": U[:, :order].flatten()},
        index=pd.MultiIndex.from_product(
            [df.index, range(order)], names=["CDF", "order"]
        ),
    )
    S_df = pd.DataFrame(
        {"coefficient": S[:order]},
        index=pd.MultiIndex.from_product([range(order)], names=["order"]),
    )
    V_df = pd.DataFrame(
        {"coefficient": Vt[:order, :].T.flatten()},
        index=pd.MultiIndex.from_product(
            [df.columns.unique(0), df.columns.unique(1), range(order)],
            names=[
                df.columns.unique(0).name,
                df.columns.unique(1).name,
                "order",
            ],
        ),
    )
    return U_df, S_df, V_df


def from_svd_dfs(
    U_df: pd.DataFrame, S_df: pd.DataFrame, V_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Reconstructs the original matrix from formatted U, S, and V DataFrames
    obtained from singular value decomposition
    """
    U = U_df["coefficient"].unstack()
    S = pd.DataFrame(
        np.diag(S_df.values.flatten()),
        index=S_df.index.get_level_values("order"),
        columns=S_df.index.get_level_values("order"),
    )
    V = V_df["coefficient"].unstack()
    return U @ S @ V.T


def print_errors(reference_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Computes the error between a reference and test DataFrame

    Parameters
    ----------
    reference_df
        The "correct" DataFrame
    test_df
        The "approximate" DataFrame
    """
    # print out normalized l2 error for DataFrame
    residuals = test_df - reference_df
    rel_frobenius_norm = np.linalg.norm(residuals, ord="fro") / np.linalg.norm(
        reference_df, ord="fro"
    )
    print(f"relative frobenius: {rel_frobenius_norm}")
    # print out absolute l-inf error for DataFrame
    abs_linf_norm = residuals.abs().max().max()
    print(f"absolute l-inf norm: {abs_linf_norm}")


