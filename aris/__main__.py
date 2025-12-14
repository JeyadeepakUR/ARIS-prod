"""
ARIS entry point module.

This module is executed when running `python -m aris`.
"""

from typing import NoReturn

from aris.logging_config import setup_logging


def main() -> NoReturn:
    """Initialize and boot the ARIS system."""
    logger = setup_logging()
    logger.info("ARIS booted")
    print("ARIS booted")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
