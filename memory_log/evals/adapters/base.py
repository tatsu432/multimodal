"""BenchmarkAdapter ABC — convert external datasets to EvalManifest format.

Implementing a new adapter:
  1. Subclass BenchmarkAdapter.
  2. Implement to_manifests() to yield EvalManifest objects.
  3. Optionally implement download() to fetch raw data.
  4. Add a __main__ block so it can be run as:
     uv run python -m evals.adapters.my_adapter --raw-dir <data> --out evals/datasets/my.json

All adapters emit EvalManifest objects, so the rest of the harness (run_live, run_ltm)
is adapter-agnostic. To switch benchmarks, just swap the manifest file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from evals.manifest import EvalManifest


class BenchmarkAdapter(ABC):
    """Convert an external benchmark into EvalManifest format.

    Args:
        raw_dir: Directory containing the downloaded benchmark files.
        limit:   Maximum number of manifests to generate (None = all).
    """

    def __init__(self, raw_dir: Path, limit: int | None = None) -> None:
        self.raw_dir = raw_dir
        self.limit = limit

    @property
    @abstractmethod
    def name(self) -> str:
        """Short benchmark name (used in manifest video_id prefix)."""

    @abstractmethod
    def to_manifests(self) -> list[EvalManifest]:
        """Convert the raw benchmark into a list of EvalManifest objects.

        Each manifest typically covers one video/scenario.
        """

    def download(self, target_dir: Path) -> None:
        """Optional: download the raw benchmark into target_dir.

        Raise NotImplementedError (default) if download must be done manually.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support automatic download. "
            f"See the adapter's docstring for manual download instructions."
        )
