#!/usr/bin/env python3
"""Houston (Houston13 -> Houston18) training entry."""
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[2] / "core"
sys.path.insert(0, str(CORE_DIR))

from trainer import build_parser, load_config_into_args, train


def main():
    default_config = Path(__file__).parent / "config.yaml"
    args = build_parser().parse_args()
    if not args.config:
        args.config = str(default_config)
    args = load_config_into_args(args, args.config)
    train(args)


if __name__ == "__main__":
    main()
