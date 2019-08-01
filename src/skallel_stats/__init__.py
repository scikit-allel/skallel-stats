# flake8: noqa
from .version import version as __version__

# Public API for the distance module.
from .distance.api import pairwise_distance

# Register backends.
from . import distance

__all__ = ['pairwise_distance']
