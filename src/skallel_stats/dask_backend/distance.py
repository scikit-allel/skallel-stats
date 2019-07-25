import numpy as np
import dask.array as da
import numba
from skallel_tensor.dask_backend import chunked_array_types, ensure_dask_array
from skallel_stats.api import distance as api


def pdist_mapper(x, *, metric, **kwargs):
    # Introduce new axis to allow for mapping blocks.
    return api.pairwise_distance(x, metric=metric, **kwargs)[None, :]


@numba.njit(nogil=True)
def hamming_mapper(x):

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


@numba.njit(nogil=True)
def jaccard_mapper(x):

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


def pairwise_distance(x, *, metric, **kwargs):

    # Check inputs.
    assert x.ndim == 2
    x = ensure_dask_array(x)

    # Rechunk across second dimension.
    x = x.rechunk((x.chunks[0], -1))

    # Compute number of blocks.
    n_blocks = len(x.chunks[0])

    # Compute number of pairs.
    n = x.shape[1]
    n_pairs = n * (n - 1) // 2

    if metric in {"cityblock", "sqeuclidean"}:

        # These are additive metrics, can just map blocks and sum.
        mapper = pdist_mapper
        mapper_kwargs = dict(metric=metric)
        chunks = ((1,) * n_blocks, (n_pairs,))
        finalizer = None

    elif metric == "euclidean":

        # Compute square euclidean in blocks, sum over blocks, then take square
        # root.
        mapper = pdist_mapper
        mapper_kwargs = dict(metric="sqeuclidean")
        chunks = ((1,) * n_blocks, (n_pairs,))
        finalizer = da.sqrt

    elif metric == "hamming":

        # Compute numerator and denominator in blocks, sum over blocks,
        # then divide.
        mapper = hamming_mapper
        mapper_kwargs = dict(new_axis=2)
        chunks = ((1,) * n_blocks, (n_pairs,), (2,))

        def finalizer(y):
            return y[:, 0] / y[:, 1]

    elif metric == "jaccard":

        # Compute numerator and denominator in blocks, sum over blocks,
        # then divide.
        mapper = jaccard_mapper
        mapper_kwargs = dict(new_axis=2)
        chunks = ((1,) * n_blocks, (n_pairs,), (2,))

        def finalizer(y):
            return y[:, 0] / y[:, 1]

    else:

        raise NotImplementedError

    # Compute distance in blocks.
    d = da.map_blocks(
        mapper, x, chunks=chunks, dtype=np.float64, **mapper_kwargs
    )

    # Sum blocks.
    out = da.sum(d, axis=0, dtype=np.float64)

    # Finalize.
    if finalizer is not None:
        out = finalizer(out)

    return out


api.dispatch_pairwise_distance.add((chunked_array_types,), pairwise_distance)
