"""Run a synthetic day of alerts through the full cascade and print the funnel.

    python scripts/demo_day.py        (same as: sentinel demo)

The implementation lives in ``sentinel/demo.py`` so it ships with the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
