import numpy as np
import pandas as pd
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from os.path import join
from pyminimc import estimator, util
from typing import Sequence


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
            if sab_file.readline()[69:75] == "1 7  4":
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
    numerators = (x[1:] - x[:-1]) * (y[1:] - y[:-1])
    log_y = np.log(y, where=y != 0)
    denominators = log_y[1:] - log_y[:-1]
    # if either value of y is zero, the logarithmic interpolated value is zero
    # throughout the interpolation range
    nonzero_bin_endpoints = np.fromiter(
        (
            y_left != 0 and y_right != 0
            for y_left, y_right in zip(y[:-1], y[1:])
        ),
        dtype=bool,
    )
    # each element is the result of integrating over each bin
    integrals = np.zeros(len(numerators))
    np.divide(
        numerators,
        denominators,
        out=integrals,
        where=nonzero_bin_endpoints & (denominators != 0),
    )
    return pd.Series(
        np.concatenate(([0], np.cumsum(integrals))),
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


def run_minimc_with_tsl(
    minimc_path: str,
    input_filepath: str,
    beta_partitions: Sequence[pd.DataFrame],
    alpha_partitions: Sequence[pd.DataFrame],
) -> estimator.NonmultiplyingBinnedLeakageEstimator:
    """
    Run a minimc problem with custom thermal scattering data

    Parameters
    ----------
    minimc_path:
        Path to minimc executable
    input_filepath:
        Path to input file
    beta_partitions:
        Beta data partitioned into incident energy ranges. Index must be CDF
        values and columns must be a MultiIndex of incident energies and
        target temperatures.
    alpha_partitions:
        Alpha data partitioned into beta ranges. Index must be CDF values and
        columns must be a MultiIndex of beta and target temperatures.
    """
    alpha_beta_partitions = {
        "beta": beta_partitions,
        "alpha": alpha_partitions,
    }
    tree = ET.parse(input_filepath)
    tsl_node_path = "nuclides/continuous/nuclide/neutron/scatter/tsl"
    if (tsl_node := tree.find(tsl_node_path)) is None:
        raise RuntimeError(f"node not found: {tsl_node_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")
        # modify input file
        for name, V_attribute in zip(["alpha", "beta"], ["beta_T", "E_T"]):
            partitions_node_name = f"{name}_partitions"
            if (
                partitions_node := tsl_node.find(partitions_node_name)
            ) is not None:
                # remove existing partitions
                tsl_node.remove(partitions_node)
            # add empty partitions node
            partitions_node = ET.Element(partitions_node_name)
            # factorize full_df and save as separate DataFrames
            for i, full_df in enumerate(alpha_beta_partitions[name]):
                partition_node = ET.Element("partition")
                for attribute, df in zip(
                    ["CDF", "S", V_attribute],
                    util.to_svd_dfs(full_df, order=10),
                ):
                    df_path = join(tmpdir, f"{name}_{i}_{attribute}.hdf5")
                    # save to temporary directory
                    df.to_hdf(df_path, "pandas")
                    # add path to DataFrame in partition node
                    partition_node.set(attribute, df_path)
                partitions_node.append(partition_node)
            tsl_node.append(partitions_node)
        with open(join(tmpdir, "minimc.inp"), "wb") as modified_inputfile:
            tree.write(modified_inputfile)
            modified_inputfile.flush()  # stackoverflow.com/a/9422590/5101335
            # run minimc and parse output
            subprocess.run([minimc_path, modified_inputfile.name])
            energy_node_path = "problemtype/fixedsource/energy/constant"
            if (energy_node := tree.find(energy_node_path)) is None or (
                energy := energy_node.get("energy")
            ) is None:
                raise RuntimeError(
                    f"missing node or attribute: {energy_node_path}"
                )
            else:
                energy = float(energy)
            temperature_node_path = "temperature/constant"
            if (
                temperature_node := tree.find(temperature_node_path)
            ) is None or (temperature := temperature_node.get("c")) is None:
                raise RuntimeError(
                    f"missing node or attribute: {temperature_node_path}"
                )
            else:
                temperature = float(temperature)
            return estimator.MiniMC(join(tmpdir, "minimc.out"))
