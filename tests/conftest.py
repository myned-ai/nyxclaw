"""Shared test fixtures and import stubs.

Stubs heavy third-party packages that are not needed for unit tests.
Must be loaded before any src/ imports that transitively pull them in.
"""

import sys
from unittest.mock import MagicMock

# Stub heavy third-party packages before any src/ code imports them.
# This avoids ImportError for packages not installed in the test venv.
_HEAVY_PACKAGES = [
    "numpy",
    "np",
    "librosa",
    "scipy",
    "scipy.signal",
    "onnxruntime",
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.asymmetric",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.serialization",
    "segno",
    "piper",
    "faster_whisper",
]

for _pkg in _HEAVY_PACKAGES:
    sys.modules.setdefault(_pkg, MagicMock())
