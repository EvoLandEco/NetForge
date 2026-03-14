"""Support unittest discovery."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "netforge-test-mpl"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
