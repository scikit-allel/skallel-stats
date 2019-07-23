import numpy as np
import dask.array as da
from skallel_tensor.dask_backend import chunked_array_types, ensure_dask_array
from skallel_stats.api import distance as api


def pdist_mapper(block, *, metric, **kwargs):
    # Introduce new axis to allow for mapping blocks.
    return api.pairwise_distance(block, metric=metric, **kwargs)[None, :]


def pairwise_distance(x, *, metric, **kwargs):

    # Check inputs.
    assert x.ndim == 2
    x = ensure_dask_array(x)

    # Rechunk across second dimension.
    x = x.rechunk((x.chunks[0], -1))

    # Compute number of pairs.
    n = x.shape[1]
    n_pairs = n * (n - 1) // 2

    if metric in {"cityblock", "sqeuclidean"}:

        # These are additive metrics, can just map blocks and sum.
        mapper_metric = metric
        finalizer = None

    elif metric == "euclidean":

        # Need to compute square euclidean in blocks, then sum, then take
        # square root.
        mapper_metric = "sqeuclidean"
        finalizer = da.sqrt

    else:

        raise NotImplementedError

    # Compute output chunks.
    chunks = ((1,) * len(x.chunks[0]), (n_pairs,))

    # Compute distance in blocks.
    d = da.map_blocks(
        pdist_mapper,
        x,
        metric=mapper_metric,
        chunks=chunks,
        dtype=np.float64,
        **kwargs
    )

    # Sum blocks.
    out = da.sum(d, axis=0, dtype=np.float64)

    # Finalize.
    if finalizer is not None:
        out = finalizer(out)

    return out


api.dispatch_pairwise_distance.add((chunked_array_types,), pairwise_distance)
