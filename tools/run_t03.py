#!/usr/bin/env python3
"""Canonical T03 preparation/execution CLI; no submission side effects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_utils.t03_runtime import main

if __name__ == '__main__':
    main()
