#!/usr/bin/env python

import pandas as pd
import pyminimc.compression
import pyminimc.util

if __name__ == "__main__":

    df = pd.read_hdf(
        "/Users/atumulak/Developer/minimc/data/tnsl/endfb8/dfs/alpha_full.hdf5"
    )

    pyminimc.compression.partitionless_coarsen(df, rank=7, log=True)
