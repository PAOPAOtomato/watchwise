"""Entry point: python -m watchwise [serve|doctor|init|check|demo|profiles] ..."""

from __future__ import annotations

import sys

from watchwise.cli import main

if __name__ == "__main__":
    sys.exit(main())
