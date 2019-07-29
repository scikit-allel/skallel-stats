import numpy as np
from numpy.testing import assert_allclose
import scipy.spatial.distance as spd
import dask.array as da
from numba import cuda
import zarr
import pytest
from skallel_stats import pairwise_distance


def _test_pairwise_distance(metric):

    # Simulate some data, N.B., oriented such that we want to compute
    # distance between columns.
    data = np.random.randint(low=0, high=3, size=(100, 10), dtype=np.int8)

    # Compute expected result, using scipy as reference implementation.
    expect = spd.pdist(data.T, metric=metric)

    # Test numpy array.
    actual = pairwise_distance(data, metric=metric)
    assert isinstance(actual, np.ndarray)
    assert_allclose(expect, actual)
    assert actual.dtype.kind == "f"

    # Test cuda array.
    data_cuda = cuda.to_device(data)
    actual = pairwise_distance(data_cuda, metric=metric)
    assert isinstance(actual, type(data_cuda))
    assert_allclose(expect, actual.copy_to_host())
    assert actual.dtype.kind == "f"

    # Test dask array.
    data_dask = da.from_array(data, chunks=(10, 5))
    actual = pairwise_distance(data_dask, metric=metric)
    assert isinstance(actual, da.Array)
    ac = actual.compute(scheduler="single-threaded")
    assert_allclose(expect, ac)
    assert actual.dtype.kind == "f"

    # Test dask array with cuda.
    data_dask_cuda = data_dask.rechunk((10, -1)).map_blocks(cuda.to_device)
    actual = pairwise_distance(data_dask_cuda, metric=metric)
    assert isinstance(actual, da.Array)
    ac = actual.compute(scheduler="single-threaded")
    assert_allclose(expect, ac)
    assert actual.dtype.kind == "f"

    # Test zarr array.
    data_zarr = zarr.array(data, chunks=(10, 5))
    actual = pairwise_distance(data_zarr, metric=metric)
    assert isinstance(actual, da.Array)
    assert_allclose(expect, actual.compute())
    assert actual.dtype.kind == "f"


def test_cityblock():
    _test_pairwise_distance("cityblock")


def test_euclidean():
    _test_pairwise_distance("euclidean")


def test_sqeuclidean():
    _test_pairwise_distance("sqeuclidean")


def test_hamming():
    _test_pairwise_distance("hamming")


def test_jaccard():
    _test_pairwise_distance("jaccard")


def test_not_implemented():

    data = np.random.randint(low=0, high=3, size=(100, 10), dtype=np.int8)
    with pytest.raises(ValueError):
        pairwise_distance(data, metric="foo")

    data_dask = da.from_array(data, chunks=(10, 5))
    with pytest.raises(ValueError):
        pairwise_distance(data_dask, metric="foo")

    data_cuda = cuda.to_device(data)
    with pytest.raises(ValueError):
        pairwise_distance(data_cuda, metric="foo")
