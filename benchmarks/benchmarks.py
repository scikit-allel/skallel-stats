import numpy as np
import dask.array as da
from numba import cuda
import os
from skallel_stats import pairwise_distance


cudasim = False
if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    cudasim = True


class TimePairwiseDistance:
    """Timing benchmarks for pairwise distance functions."""

    def setup(self):
        self.data = np.random.randint(-1, 4, size=(20000, 1000, 2), dtype="i1")
        self.data_dask = da.from_array(self.data, chunks=(2000, 1000, 2))
        if not cudasim:
            self.data_cuda = cuda.to_device(self.data)
            self.data_dask_cuda = self.data_dask.map_blocks(cuda.to_device)

    def time_pairwise_distance(self):
        pairwise_distance(self.data)
