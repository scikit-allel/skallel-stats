import numpy as np
import dask.array as da
from skallel_tensor.dask_backend import chunked_array_types, ensure_dask_array
from . import api


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

    if metric == "cityblock":

        # Compute distance in blocks.
        chunks = ((1,) * n_blocks, (n_pairs,))
        d = da.map_blocks(
            api.dispatch_map_block_cityblock, x, chunks=chunks, dtype=np.float64
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        return out

    elif metric == "sqeuclidean":

        # Compute distance in blocks.
        chunks = ((1,) * n_blocks, (n_pairs,))
        d = da.map_blocks(
            api.dispatch_map_block_sqeuclidean,
            x,
            chunks=chunks,
            dtype=np.float64,
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        return out

    elif metric == "euclidean":

        # Compute distance in blocks.
        chunks = ((1,) * n_blocks, (n_pairs,))
        d = da.map_blocks(
            api.dispatch_map_block_sqeuclidean,
            x,
            chunks=chunks,
            dtype=np.float64,
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        # Finalize.
        out = da.sqrt(out)

        return out

    elif metric == "hamming":

        # Compute distance in blocks.
        chunks = ((1,) * n_blocks, (n_pairs,), (2,))
        d = da.map_blocks(
            api.dispatch_map_block_hamming,
            x,
            chunks=chunks,
            dtype=np.float64,
            new_axis=2,
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        # Finalize.
        out = out[:, 0] / out[:, 1]

        return out

    elif metric == "jaccard":

        # Compute distance in blocks.
        chunks = ((1,) * n_blocks, (n_pairs,), (2,))
        d = da.map_blocks(
            api.dispatch_map_block_jaccard,
            x,
            chunks=chunks,
            dtype=np.float64,
            new_axis=2,
        )

        # Sum blocks.
        out = da.sum(d, axis=0, dtype=np.float64)

        # Finalize.
        out = out[:, 0] / out[:, 1]

        return out

    else:

        raise ValueError


api.dispatch_pairwise_distance.add((chunked_array_types,), pairwise_distance)
