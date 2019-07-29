import numpy as np
import dask.array as da
from numba import cuda
import os
from skallel_stats import pairwise_distance


class TimePairwiseDistanceNumpy:
    """Timing benchmarks for pairwise distance functions."""

    def setup(self):
        self.data = np.random.randint(
            low=0, high=3, size=(2000, 1000), dtype=np.int8
        )

    def time_cityblock(self):
        pairwise_distance(self.data, metric="cityblock")

    def time_sqeuclidean(self):
        pairwise_distance(self.data, metric="sqeuclidean")

    def time_euclidean(self):
        pairwise_distance(self.data, metric="euclidean")

    def time_hamming(self):
        pairwise_distance(self.data, metric="hamming")

    def time_jaccard(self):
        pairwise_distance(self.data, metric="jaccard")


class TimePairwiseDistanceDask:
    """Timing benchmarks for pairwise distance functions."""

    def setup(self):
        self.data = np.random.randint(
            low=0, high=3, size=(2000, 1000), dtype=np.int8
        )
        self.data_dask = da.from_array(self.data, chunks=(200, -1))

    def time_cityblock(self):
        pairwise_distance(self.data_dask, metric="cityblock").compute()

    def time_sqeuclidean(self):
        pairwise_distance(self.data_dask, metric="sqeuclidean").compute()

    def time_euclidean(self):
        pairwise_distance(self.data_dask, metric="euclidean").compute()

    def time_hamming(self):
        pairwise_distance(self.data_dask, metric="hamming").compute()

    def time_jaccard(self):
        pairwise_distance(self.data_dask, metric="jaccard").compute()


class TimePairwiseDistanceCuda:
    """Timing benchmarks for pairwise distance functions."""

    def setup(self):
        if (
            not cuda.is_available()
            or os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"
        ):
            raise NotImplementedError

        self.data = np.random.randint(
            low=0, high=3, size=(2000, 1000), dtype=np.int8
        )
        self.data_cuda = cuda.to_device(self.data)

    def time_cityblock(self):
        pairwise_distance(self.data_cuda, metric="cityblock")
        cuda.synchronize()

    def time_sqeuclidean(self):
        pairwise_distance(self.data_cuda, metric="sqeuclidean")
        cuda.synchronize()

    def time_euclidean(self):
        pairwise_distance(self.data_cuda, metric="euclidean")
        cuda.synchronize()

    def time_hamming(self):
        pairwise_distance(self.data_cuda, metric="hamming")
        cuda.synchronize()

    def time_jaccard(self):
        pairwise_distance(self.data_cuda, metric="jaccard")
        cuda.synchronize()
