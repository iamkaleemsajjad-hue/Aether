"""Aether CLI entrypoint for `python -m aether`."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for running `python -m aether`."""
    from aether.cli import cli

    sys.exit(cli())


if __name__ == "__main__":
    main()
