import numpy as np
from numpy.testing import assert_allclose
import scipy.spatial.distance as spd
import dask.array as da
import zarr
from skallel_stats import pairwise_distance


def test_pairwise_distance_cityblock():
    metric = "cityblock"

    # Simulate some data, N.B., oriented such that we want to compute
    # distance between columns.
    data = np.random.randint(low=0, high=3, size=(100, 10), dtype=np.int8)

    # Compute expected result, using scipy as reference implementation.
    expect = spd.pdist(data.T, metric=metric)

    # Test numpy array.
    actual = pairwise_distance(data, metric=metric)
    assert isinstance(actual, np.ndarray)
    assert_allclose(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(10, 5))
    actual = pairwise_distance(data_dask, metric=metric)
    assert isinstance(actual, da.Array)
    ac = actual.compute()
    assert_allclose(expect, ac)
    assert expect.dtype == actual.dtype

    # Test zarr array.
    data_zarr = zarr.array(data, chunks=(10, 5))
    actual = pairwise_distance(data_zarr, metric=metric)
    assert isinstance(actual, da.Array)
    assert_allclose(expect, actual.compute())
    assert expect.dtype == actual.dtype