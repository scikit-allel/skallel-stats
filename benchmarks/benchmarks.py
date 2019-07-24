import numpy as np
import dask.array as da

# from numba import cuda
import os
from skallel_stats import pairwise_distance


cudasim = False
if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    cudasim = True


class TimePairwiseDistance:
    """Timing benchmarks for pairwise distance functions."""

    def setup(self):
        self.data = np.random.randint(
            low=0, high=3, size=(20000, 100), dtype=np.int8
        )
        self.data_dask = da.from_array(self.data, chunks=(2000, -1))
        # if not cudasim:
        #     self.data_cuda = cuda.to_device(self.data)
        #     self.data_dask_cuda = self.data_dask.map_blocks(cuda.to_device)

    def time_cityblock_numpy(self):
        pairwise_distance(self.data, metric="cityblock")

    def time_cityblock_dask(self):
        pairwise_distance(self.data_dask, metric="cityblock").compute()

    def time_sqeuclidean_numpy(self):
        pairwise_distance(self.data, metric="sqeuclidean")

    def time_sqeuclidean_dask(self):
        pairwise_distance(self.data_dask, metric="sqeuclidean").compute()

    def time_euclidean_numpy(self):
        pairwise_distance(self.data, metric="euclidean")

    def time_euclidean_dask(self):
        pairwise_distance(self.data_dask, metric="euclidean").compute()

    def time_hamming_numpy(self):
        pairwise_distance(self.data, metric="hamming")

    def time_hamming_dask(self):
        pairwise_distance(self.data_dask, metric="hamming").compute()

    def time_jaccard_numpy(self):
        pairwise_distance(self.data, metric="jaccard")

    def time_jaccard_dask(self):
        pairwise_distance(self.data_dask, metric="jaccard").compute()
