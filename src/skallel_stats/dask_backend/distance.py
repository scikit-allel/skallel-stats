import numpy as np
import dask.array as da
from skallel_tensor.dask_backend import chunked_array_types, ensure_dask_array
from skallel_stats.api import distance as api


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
            api.pairwise_distance,
            x,
            chunks=chunks,
            metric=metric,
            dtype=np.float64,
        )
        print(d.shape)

        # Sum blocks
        out = d.sum(axis=0)

        return out

    else:
        raise NotImplementedError


api.dispatch_pairwise_distance.add((chunked_array_types,), pairwise_distance)
