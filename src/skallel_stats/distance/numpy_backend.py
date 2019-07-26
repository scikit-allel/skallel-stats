import numpy as np
import numba
import scipy.spatial.distance as spd
from . import api


def pairwise_distance(x, *, metric, **kwargs):
    assert x.ndim == 2
    out = spd.pdist(x.T, metric=metric, **kwargs)
    return out


api.dispatch_pairwise_distance.add((np.ndarray,), pairwise_distance)


def map_block_cityblock(x):
    assert x.ndim == 2
    out = spd.pdist(x.T, metric="cityblock")
    # Introduce new axis to allow for mapping blocks.
    return out[None, :]


api.dispatch_map_block_cityblock.add((np.ndarray,), map_block_cityblock)


def map_block_sqeuclidean(x):
    assert x.ndim == 2
    out = spd.pdist(x.T, metric="sqeuclidean")
    # Introduce new axis to allow for mapping blocks.
    return out[None, :]


api.dispatch_map_block_sqeuclidean.add((np.ndarray,), map_block_sqeuclidean)


@numba.njit(nogil=True)
def map_block_hamming(x):
    assert x.ndim == 2

    # Dimensions.
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = n * (n - 1) // 2

    # Set up outputs.
    num = np.zeros((1, n_pairs), dtype=np.float64)
    den = np.full((1, n_pairs), m, dtype=np.float64)

    # Iterate over data.
    for i in range(m):
        pair_index = 0
        for j in range(n):
            u = x[i, j]
            for k in range(j + 1, n):
                v = x[i, k]
                if u != v:
                    num[0, pair_index] += 1
                pair_index += 1

    # Stack outputs for single return value.
    out = np.dstack((num, den))
    return out


api.dispatch_map_block_hamming.add((np.ndarray,), map_block_hamming)


@numba.njit(nogil=True)
def map_block_jaccard(x):
    assert x.ndim == 2

    # Dimensions.
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = n * (n - 1) // 2

    # Set up outputs.
    num = np.zeros((1, n_pairs), dtype=np.float64)
    den = np.zeros((1, n_pairs), dtype=np.float64)

    # Iterate over data.
    for i in range(m):
        pair_index = 0
        for j in range(n):
            u = x[i, j]
            for k in range(j + 1, n):
                v = x[i, k]
                if u > 0 or v > 0:
                    den[0, pair_index] += 1
                    if u != v:
                        num[0, pair_index] += 1
                pair_index += 1

    # Stack outputs for single return value.
    out = np.dstack((num, den))
    return out


api.dispatch_map_block_jaccard.add((np.ndarray,), map_block_jaccard)
