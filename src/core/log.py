"""Project logging

Stdlib-shadow note: this module is literally named ``logging``. The bare
``import logging`` below still resolves to the real stdlib because within the
package the module path is ``src.core.logging`` and Python 3 uses absolute
imports -- ``import logging`` is the top-level stdlib, not this file. (A relative
``from . import logging`` would be the self-reference; we never do that.)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure root logging with a stdout handler and an optional file handler.

    Idempotent: existing handlers are cleared first so repeated calls (e.g. in a
    notebook or across script re-runs) don't duplicate every line. Pass
    ``log_file`` to also tee output to a per-run logfile. Returns the project
    logger; modules should use ``get_logger(__name__)``.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger("cse164")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger under the ``cse164`` namespace."""
    return logging.getLogger(f"cse164.{name}" if name else "cse164")


__all__ = ["setup_logging", "get_logger", "LOG_FORMAT", "LOG_DATEFMT"]
