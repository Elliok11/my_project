"""让 `python -m scheduler` 可用。"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
