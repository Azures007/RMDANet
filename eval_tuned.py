#!/usr/bin/env python3
"""Run scripts/final_eval_locked.py with the DRSE runtime patch applied,
so tuned checkpoints (e.g. residual alpha) load strictly.

Use the SAME DRSE_* environment variables as the training run.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))

import model  # noqa: E402
import drse_patch  # noqa: E402

drse_patch.patch(model)

runpy.run_path(str(ROOT / "scripts" / "final_eval_locked.py"), run_name="__main__")
