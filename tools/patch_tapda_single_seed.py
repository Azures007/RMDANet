"""Restrict TAPDA's seed loop to the first N seeds (default 1).

TAPDA runs all datasets/seeds defined by the module-level ``SEEDS`` constant:

    for dataset_index, seed in enumerate(SEEDS):

This patch slices that constant so that only the first ``--num-seeds`` entries
are used, which makes single-seed ablations fast while keeping the original
protocol available (``--num-seeds 10`` restores it).  Idempotent.

Usage:
    python tools/patch_tapda_single_seed.py --tapda-dir /root/autodl-tmp/tapda --num-seeds 1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

LOOP_PATTERN = re.compile(r"enumerate\(\s*(?P<name>SEEDS|seeds|SEED_LIST|seed_list)\s*\)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapda-dir", default="/root/autodl-tmp/tapda")
    parser.add_argument("--script", default="up2pc.py")
    parser.add_argument("--num-seeds", type=int, default=1)
    options = parser.parse_args()

    path = Path(options.tapda_dir) / options.script
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf8")

    match = LOOP_PATTERN.search(text)
    if match is None:
        raise RuntimeError(
            "seed loop 'enumerate(SEEDS)' not found; run: grep -n enumerate up2pc.py"
        )
    name = match.group("name")
    replacement = f"enumerate({name}[:{options.num_seeds}])"
    if replacement in text:
        print(f"already limited to {options.num_seeds} seed(s):", replacement)
        return

    text = text[: match.start()] + replacement + text[match.end():]
    path.write_text(text, encoding="utf8")
    print(f"patched: {path}")
    print(f" - seed loop is now: for ... in {replacement}")
    for number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if replacement in line:
            print(f"   line {number}: {line.strip()}")
    print("OK")


if __name__ == "__main__":
    main()
