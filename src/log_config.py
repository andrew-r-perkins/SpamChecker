"""
log_config.py — Centralised logging configuration for all CLI entry points.

Verbosity levels (set via command-line flags):
  (no flag)  WARNING  — only problems that need attention
  -v         INFO     — normal operational milestones (startup, results)
  -vv        DEBUG    — detailed diagnostics: shapes, sizes, timings
  -vvv       TRACE    — very verbose internals: feature vectors, top TF-IDF terms

File logging:
  Every run writes a timestamped log file to LOG_DIR (default: logs/).
  The file always captures TRACE and above regardless of console verbosity,
  so full detail is available for post-mortem debugging.
  Files are rotated at LOG_MAX_BYTES; LOG_BACKUP_COUNT rotated copies are
  kept per session. On startup, old session files beyond LOG_MAX_SESSIONS
  are pruned (oldest first, including their rotated siblings).

Usage in any CLI entry point:
    import argparse
    from src.log_config import add_verbosity_args, setup_logging

    parser = argparse.ArgumentParser(...)
    add_verbosity_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)

Then use logger.trace(...) anywhere for TRACE-level output, or
logger.debug / .info / .warning as normal.
"""

import argparse
import glob
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------------
# TRACE — custom level below DEBUG (10) for very granular internals output.
# Registered once when this module is first imported.
# ---------------------------------------------------------------------------
TRACE = 5
logging.addLevelName(TRACE, 'TRACE')


def _trace(self, message, *args, **kwargs):
    """Convenience method mirroring Logger.debug() for the TRACE level."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Patch Logger once — every logger in the process gains .trace()
if not hasattr(logging.Logger, 'trace'):
    logging.Logger.trace = _trace


# ---------------------------------------------------------------------------
# Verbosity count → logging level
# ---------------------------------------------------------------------------
_VERBOSITY_MAP = {
    0: logging.WARNING,   # default — quiet, only problems
    1: logging.INFO,      # -v
    2: logging.DEBUG,     # -vv
    3: TRACE,             # -vvv
}

_LOG_FORMAT  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# File logging configuration — edit these constants to change behaviour.
# ---------------------------------------------------------------------------
LOG_DIR          = "logs"           # directory for log files (relative to project root)
LOG_MAX_BYTES    = 5 * 1024 * 1024  # 5 MB — rotate when a log file exceeds this size
LOG_BACKUP_COUNT = 3                # rotated files kept per session (.log.1 / .2 / .3)
LOG_MAX_SESSIONS = 10               # max session log files kept; oldest are deleted on startup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prune_old_logs() -> None:
    """Delete oldest session log files (and their rotated siblings) so that
    after the new session file is created there are at most LOG_MAX_SESSIONS
    base files in LOG_DIR."""
    # Base files only (e.g. api_20260303_143022.log) — glob won't match .log.1 etc.
    base_files = sorted(glob.glob(os.path.join(LOG_DIR, "api_*.log")))

    # Allow (LOG_MAX_SESSIONS - 1) existing files so the new session fits within the cap
    excess = len(base_files) - (LOG_MAX_SESSIONS - 1)
    if excess <= 0:
        return

    for base in base_files[:excess]:
        # Remove the base file and any rotated siblings (.log.1, .log.2, ...)
        for path in glob.glob(base + "*"):
            try:
                os.remove(path)
            except OSError:
                pass  # already gone or permission issue — safe to ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds -v / --verbose to an ArgumentParser.

    Each -v increments the counter:
      -v  → verbosity=1 (INFO)
      -vv → verbosity=2 (DEBUG)    (pass -v twice, or use -vv)
      -vvv→ verbosity=3 (TRACE)
    """
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help=(
            'Increase log verbosity (stackable). '
            '-v INFO | -vv DEBUG | -vvv TRACE'
        ),
    )


def setup_logging(verbosity: int) -> int:
    """
    Configures the root logger with a console handler and a rotating file handler.

    Console handler: level controlled by verbosity flag (WARNING by default).
    File handler:    always TRACE — captures everything for post-mortem debugging.

    Safe to call multiple times — does nothing if handlers are already configured.

    Args:
        verbosity: integer (0–3+). Values above 3 are clamped to TRACE.

    Returns:
        The resolved console logging level integer (e.g. logging.INFO = 20).
    """
    root = logging.getLogger()

    # Guard: basicConfig / previous call already configured handlers
    if root.handlers:
        return _VERBOSITY_MAP.get(min(verbosity, 3), TRACE)

    console_level = _VERBOSITY_MAP.get(min(verbosity, 3), TRACE)
    formatter     = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # ── File handler ───────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    _prune_old_logs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"api_{timestamp}.log")

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(TRACE)   # capture everything — file is the full record
    file_handler.setFormatter(formatter)

    # ── Root logger ────────────────────────────────────────────────────────
    # Set root level to the lowest of the two handlers so neither is silenced
    # at the logger level before messages reach the handlers.
    root.setLevel(min(console_level, TRACE))
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Verbosity=%d → console level: %s | log file: %s",
        verbosity,
        logging.getLevelName(console_level),
        log_path,
    )
    return console_level
