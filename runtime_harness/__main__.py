"""Allow `python -m runtime_harness` alongside the `rex` console script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
