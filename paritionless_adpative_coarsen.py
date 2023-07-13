#!/usr/bin/env python

import pandas as pd
from pyminimc.compression import partitionless_adaptive_coarsen
import multiprocessing
import os

if __name__ == "__main__":
    # vaguely related to https://stackoverflow.com/q/38837948/5101335
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    multiprocessing.freeze_support()

    # df = pd.read_hdf("./coarse_df.hdf5")
    df = pd.read_hdf(
        "/Users/atumulak/Developer/minimc/data/tnsl/endfb8/dfs/alpha_full.hdf5"
    )

    partitionless_adaptive_coarsen(
        df,
        cutoff=1636.7475317348378,
    )
