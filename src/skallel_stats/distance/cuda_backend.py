import numpy as np
from numba import cuda
from . import api
import math

# Simulated CUDA arrays.
from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray

cuda_array_types = (FakeCUDAArray,)
try:  # pragma: no cover
    # noinspection PyUnresolvedReferences
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    cuda_array_types += (DeviceNDArray,)
except ImportError:
    # Not available when using CUDA simulator.
    pass


def pairwise_distance(x, *, metric, **kwargs):
    assert x.ndim == 2

    # Set up output array.
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    out = cuda.device_array(n_pairs, dtype=np.float32)

    if metric == "cityblock":
        # Let numba decide number of threads and blocks.
        kernel = kernel_cityblock.forall(n_pairs)
        kernel(x, out)
        return out

    else:
        raise NotImplementedError


api.dispatch_pairwise_distance.add((cuda_array_types,), pairwise_distance)


@cuda.jit(device=True)
def square_coords(pair_index, n):
    pair_index = np.float32(pair_index)
    n = np.float32(n)
    j = (((2 * n) - 1) - math.sqrt((1 - (2 * n)) ** 2 - (8 * pair_index))) // 2
    k = pair_index - (j * ((2 * n) - j - 1) / 2) + j + 1
    j = np.int64(j)
    k = np.int64(k)
    return j, k


@cuda.jit
def kernel_cityblock(x, out):
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    pair_index = cuda.grid(1)
    if pair_index < n_pairs:
        # Unpack the pair index to column indices.
        j, k = square_coords(pair_index, n)
        # Iterate over rows, accumulating distance.
        d = np.float32(0)
        for i in range(m):
            u = np.float32(x[i, j])
            v = np.float32(x[i, k])
            d += math.fabs(u - v)
        # Store distance result.
        out[pair_index] = d
