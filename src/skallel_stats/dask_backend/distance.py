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

    if metric == "cityblock":

        # Compute output chunks.
        chunks = ((1,) * len(x.chunks[0]), (n_pairs,))

        # Compute distance in blocks.
        d = da.map_blocks(
            pdist_mapper,
            x,
            metric=metric,
            chunks=chunks,
            dtype=np.float64,
            **kwargs
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        return out

    else:
        raise NotImplementedError


api.dispatch_pairwise_distance.add((chunked_array_types,), pairwise_distance)
