#!/usr/bin/env python3
"""Legacy evaluation entry.

The residual DRSE is now baked into ``core/model.py`` (``CenterGuidedPooling``
with ``residual=True`` and a learnable ``_alpha_logit``), so normal checkpoints
load with ``scripts/final_eval_locked.py`` directly.

Use this wrapper ONLY for checkpoints trained with the old runtime patch
(set ``DRSE_LEGACY=1``), which are missing/extra-keyed relative to the baked
model.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))

if os.environ.get("DRSE_LEGACY", "0") == "1":
    import model  # noqa: E402
    import drse_patch  # noqa: E402

    drse_patch.patch(model)
    print("[legacy] DRSE runtime patch applied", flush=True)

runpy.run_path(str(ROOT / "scripts" / "final_eval_locked.py"), run_name="__main__")
