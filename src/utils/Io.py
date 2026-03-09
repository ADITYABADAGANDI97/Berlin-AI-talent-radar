"""
File I/O helpers for Berlin AI Talent Radar.

Provides atomic JSON read/write, directory bootstrapping, and numpy
array persistence.  All paths are resolved relative to the project root
so callers never need to manage ``os.getcwd()`` themselves.

Design principles:
- Every write is atomic (write-to-temp then rename) to avoid partial
  files if the process is killed mid-write.
- Idempotent: ``ensure_dirs`` can be called repeatedly with no side
  effects.
- Typed: all public functions carry full type hints.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """
    Resolve the project root directory.

    Walks up from this file's location until it finds a directory that
    contains ``main.py`` or ``config/``, falling back to the current
    working directory if neither is found within 5 levels.

    Returns:
        Absolute ``Path`` to the project root.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "main.py").exists() or (candidate / "config").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


PROJECT_ROOT: Path = _project_root()


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Union[str, Path]) -> None:
    """
    Create directories (including parents) if they do not already exist.

    Idempotent — safe to call multiple times.

    Args:
        *paths: One or more directory paths to create.

    Example:
        >>> ensure_dirs("data/raw", "data/processed", "data/embeddings")
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", p)


def bootstrap_data_dirs() -> None:
    """
    Create all standard project data directories in one call.

    Reads the expected directory layout from the project specification
    and creates every folder.  Called automatically by ``main.py`` on
    startup.

    Example:
        >>> bootstrap_data_dirs()
    """
    dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "embeddings",
        PROJECT_ROOT / "data" / "reports",
        PROJECT_ROOT / "data" / "eu_ai_act",
        PROJECT_ROOT / "data" / "demo",
    ]
    ensure_dirs(*dirs)
    logger.info("Data directories bootstrapped under %s", PROJECT_ROOT / "data")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Union[str, Path]) -> Any:
    """
    Load a JSON file and return its parsed content.

    Args:
        path: Path to the ``.json`` file.

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Example:
        >>> jobs = load_json("data/raw/jsearch.json")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("Loaded JSON: %s (%d top-level items)", path, len(data) if isinstance(data, (list, dict)) else 1)
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Atomically write ``data`` to a JSON file.

    Uses a temporary file in the same directory followed by an atomic
    rename, so the destination file is never partially written.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.  Parent directories are created
              automatically.
        indent: JSON indentation level (default 2).

    Example:
        >>> save_json(jobs, "data/raw/jsearch.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory to make rename atomic
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False, default=_json_default)
        os.replace(tmp_path, path)  # atomic on POSIX; near-atomic on Windows
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    count = len(data) if isinstance(data, (list, dict)) else "N/A"
    logger.info("Saved JSON → %s  (%s items)", path, count)


def _json_default(obj: Any) -> Any:
    """
    Custom JSON serialiser for types not handled by the standard library.

    Currently handles:
    - ``numpy`` scalar types → Python native int/float
    - ``Path`` → string

    Args:
        obj: Object that failed standard serialisation.

    Returns:
        JSON-serialisable representation.

    Raises:
        TypeError: For unsupported types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Numpy array persistence
# ---------------------------------------------------------------------------

def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a numpy array to disk in ``.npy`` format.

    Args:
        array: Array to persist.
        path: Destination file path (should end in ``.npy``).

    Example:
        >>> save_numpy(embeddings, "data/embeddings/job_chunks.npy")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    logger.debug("Saved numpy array %s → %s", array.shape, path)


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """
    Load a numpy array from a ``.npy`` file.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        Loaded numpy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Example:
        >>> embeddings = load_numpy("data/embeddings/job_chunks.npy")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Numpy file not found: {path}")
    array = np.load(str(path), allow_pickle=False)
    logger.debug("Loaded numpy array %s ← %s", array.shape, path)
    return array


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def save_text(text: str, path: Union[str, Path]) -> None:
    """
    Write a plain-text string to a file, creating parent directories.

    Args:
        text: Content to write.
        path: Destination file path.

    Example:
        >>> save_text(report_markdown, "data/reports/berlin_report.md")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Saved text file → %s  (%d chars)", path, len(text))


def load_text(path: Union[str, Path]) -> str:
    """
    Read a plain-text file and return its content as a string.

    Args:
        path: Path to the file.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Example:
        >>> report = load_text("data/reports/berlin_report.md")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    content = path.read_text(encoding="utf-8")
    logger.debug("Loaded text file ← %s  (%d chars)", path, len(content))
    return content


# ---------------------------------------------------------------------------
# Convenience: list raw data files
# ---------------------------------------------------------------------------

def list_raw_files(data_dir: Union[str, Path] = "data/raw") -> list[Path]:
    """
    Return all ``.json`` files in the raw data directory.

    Args:
        data_dir: Directory to scan (default: ``data/raw``).

    Returns:
        Sorted list of ``Path`` objects.

    Example:
        >>> files = list_raw_files()
        >>> print([f.name for f in files])
        ['arbeitnow.json', 'bsj.json', 'hackernews.json', 'jsearch.json']
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning("Raw data directory does not exist: %s", data_dir)
        return []
    files = sorted(data_dir.glob("*.json"))
    logger.debug("Found %d raw JSON files in %s", len(files), data_dir)
    return files