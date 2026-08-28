"""
DeepChem Server Client Package

A Python client for interacting with the DeepChem server API.
Provides Primitives client for computation tasks and Data client for data operations.
"""

__version__ = "0.1.0"

from .base import BaseClient
from .data import Data
from .primitives.base import Primitive
from .primitives.del_denoising import DelDenoise
from .primitives.evaluate import Evaluate
from .primitives.featurize import Featurize
from .primitives.docking import Docking
from .primitives.infer import Infer
from .primitives.partition import Partition
from .primitives.splitter import TVTSplit
from .primitives.train import Train
from .primitives.transform import Transform
from .settings import Settings


__all__ = [
    "Settings",
    "Data",
    "BaseClient",
    "Primitive",
    "DelDenoise",
    "Featurize",
    "Docking",
    "Train",
    "Evaluate",
    "Infer",
    "Partition",
    "TVTSplit",
    "Transform",
]
