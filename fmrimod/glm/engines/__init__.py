"""Built-in fitting engines.

Engines are auto-registered when this subpackage is imported. The import
is triggered lazily from ``fmrimod.glm.engine`` on first engine lookup.
"""

from .runwise import RunwiseEngine
from .sketch import SketchEngine
from .chunkwise import ChunkwiseEngine
from .concat import ConcatEngine

__all__ = ["RunwiseEngine", "SketchEngine", "ChunkwiseEngine", "ConcatEngine"]
