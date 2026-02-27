"""
log_config.py — Centralised logging configuration for all CLI entry points.

Verbosity levels (set via command-line flags):
  (no flag)  WARNING  — only problems that need attention
  -v         INFO     — normal operational milestones (startup, results)
  -vv        DEBUG    — detailed diagnostics: shapes, sizes, timings
  -vvv       TRACE    — very verbose internals: feature vectors, top TF-IDF terms

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

import logging
import argparse

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
    Configures the root logger based on the verbosity count from argparse.

    Args:
        verbosity: integer (0–3+). Values above 3 are clamped to TRACE.

    Returns:
        The resolved logging level integer (e.g. logging.DEBUG = 10).
    """
    level = _VERBOSITY_MAP.get(min(verbosity, 3), TRACE)

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    # This line itself is only visible at INFO or below
    logging.getLogger(__name__).info(
        "Verbosity=%d → log level: %s",
        verbosity,
        logging.getLevelName(level),
    )
    return level
