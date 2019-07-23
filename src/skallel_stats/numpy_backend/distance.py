import numpy as np
import scipy.spatial.distance as spd
from skallel_stats.api import distance as api


def pairwise_distance(x, *, metric, **kwargs):
    assert x.ndim == 2
    out = spd.pdist(x.T, metric=metric, **kwargs)
    return out


api.dispatch_pairwise_distance.add((np.ndarray,), pairwise_distance)
