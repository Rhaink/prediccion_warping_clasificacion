#!/usr/bin/env python3
"""
Wrapper for `python -m src_v2 train-classifier`.

Keeps backward compatibility while the CLI remains the source of truth.
"""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    cmd = [sys.executable, "-m", "src_v2", "train-classifier", *sys.argv[1:]]
    env = os.environ.copy()
    result = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
