from multipledispatch import Dispatcher
from skallel_tensor.utils import check_array_like


def pairwise_distance(x, *, metric, **kwargs):
    """TODO"""

    # Check inputs.
    check_array_like(x, ndim=2)

    # Dispatch.
    return dispatch_pairwise_distance(x, metric=metric, **kwargs)


dispatch_pairwise_distance = Dispatcher("dispatch_pairwise_distance")
