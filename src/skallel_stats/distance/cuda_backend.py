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
try:  # pragma: no cover
    import cupy

    cuda_array_types += (cupy.ndarray,)
except ImportError:
    # For testing, using numpy as cupy.
    import numpy as cupy


def pairwise_distance(x, *, metric, **kwargs):
    assert x.ndim == 2

    if metric == "cityblock":
        kernel = kernel_cityblock

    elif metric == "sqeuclidean":
        kernel = kernel_sqeuclidean

    elif metric == "euclidean":
        kernel = kernel_euclidean

    elif metric == "hamming":
        kernel = kernel_hamming

    elif metric == "jaccard":
        kernel = kernel_jaccard

    else:
        raise ValueError

    # Set up output array.
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    out = cuda.device_array(n_pairs, dtype=np.float32)

    # Let numba decide number of threads and blocks.
    kernel_spec = kernel.forall(n_pairs)
    kernel_spec(x, out)

    return out


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


@cuda.jit
def kernel_sqeuclidean(x, out):
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
            d += (u - v) ** 2
        # Store distance result.
        out[pair_index] = d


@cuda.jit
def kernel_euclidean(x, out):
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
            d += (u - v) ** 2
        # Store distance result.
        out[pair_index] = math.sqrt(d)


@cuda.jit
def kernel_hamming(x, out):
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    pair_index = cuda.grid(1)
    if pair_index < n_pairs:
        # Unpack the pair index to column indices.
        j, k = square_coords(pair_index, n)
        # Iterate over rows, accumulating distance.
        numerator = np.float32(0)
        for i in range(m):
            u = x[i, j]
            v = x[i, k]
            numerator += u != v
        # Store distance result.
        out[pair_index] = numerator / m


@cuda.jit
def kernel_jaccard(x, out):
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    pair_index = cuda.grid(1)
    if pair_index < n_pairs:
        # Unpack the pair index to column indices.
        j, k = square_coords(pair_index, n)
        # Iterate over rows, accumulating distance.
        numerator = np.float32(0)
        denominator = np.float32(0)
        for i in range(m):
            u = x[i, j]
            v = x[i, k]
            t = u > 0 or v > 0
            denominator += t
            numerator += t and u != v
        # Store distance result.
        out[pair_index] = numerator / denominator


def map_block_cityblock(x):
    assert x.ndim == 2
    out = pairwise_distance(x, metric="cityblock")
    out = out.reshape((1, out.shape[0]))
    return out


api.dispatch_map_block_cityblock.add((cuda_array_types,), map_block_cityblock)


def map_block_sqeuclidean(x):
    assert x.ndim == 2
    out = pairwise_distance(x, metric="sqeuclidean")
    out = out.reshape((1, out.shape[0]))
    return out


api.dispatch_map_block_sqeuclidean.add(
    (cuda_array_types,), map_block_sqeuclidean
)


def map_block_hamming(x):
    assert x.ndim == 2
    # Set up output array.
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    out = cuda.device_array((1, n_pairs, 2), dtype=np.float32)
    # Let numba decide number of threads and blocks.
    kernel_spec = kernel_map_block_hamming.forall(n_pairs)
    kernel_spec(x, out)
    # Convert to cupy array for further work on GPU.
    out = cupy.asarray(out)
    return out


api.dispatch_map_block_hamming.add((cuda_array_types,), map_block_hamming)


@cuda.jit
def kernel_map_block_hamming(x, out):
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    pair_index = cuda.grid(1)
    if pair_index < n_pairs:
        # Unpack the pair index to column indices.
        j, k = square_coords(pair_index, n)
        # Iterate over rows, accumulating distance.
        numerator = np.float32(0)
        for i in range(m):
            u = x[i, j]
            v = x[i, k]
            numerator += u != v
        # Store distance result.
        out[0, pair_index, 0] = numerator
        out[0, pair_index, 1] = m


def map_block_jaccard(x):
    assert x.ndim == 2
    # Set up output array.
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    out = cuda.device_array((1, n_pairs, 2), dtype=np.float32)
    # Let numba decide number of threads and blocks.
    kernel_spec = kernel_map_block_jaccard.forall(n_pairs)
    kernel_spec(x, out)
    # Convert to cupy array for further work on GPU.
    out = cupy.asarray(out)
    return out


api.dispatch_map_block_jaccard.add((cuda_array_types,), map_block_jaccard)


@cuda.jit
def kernel_map_block_jaccard(x, out):
    m = x.shape[0]
    n = x.shape[1]
    n_pairs = (n * (n - 1)) // 2
    pair_index = cuda.grid(1)
    if pair_index < n_pairs:
        # Unpack the pair index to column indices.
        j, k = square_coords(pair_index, n)
        # Iterate over rows, accumulating distance.
        numerator = np.float32(0)
        denominator = np.float32(0)
        for i in range(m):
            u = x[i, j]
            v = x[i, k]
            t = u > 0 or v > 0
            denominator += t
            numerator += t and u != v
        # Store distance result.
        out[0, pair_index, 0] = numerator
        out[0, pair_index, 1] = denominator
