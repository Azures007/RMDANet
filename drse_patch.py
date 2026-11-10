"""Runtime tuning patch for CenterGuidedPooling (DRSE).

Controlled by env vars; no-op unless DRSE_TUNE=1.

  DRSE_TUNE=1
  DRSE_CENTER_SIZE=5          # default 3
  DRSE_TEMP=0.30              # default 0.12
  DRSE_INIT=0.6,0.3,0.1       # initial softmax weights: global,center,attended
"""
import os

import torch

ENABLED = os.environ.get("DRSE_TUNE", "0") == "1"
CENTER_SIZE = int(os.environ.get("DRSE_CENTER_SIZE", "5"))
TEMP = float(os.environ.get("DRSE_TEMP", "0.30"))
INIT = tuple(float(x) for x in os.environ.get("DRSE_INIT", "0.6,0.3,0.1").split(","))
assert len(INIT) == 3, "DRSE_INIT must be three comma-separated weights"


def patch(model_module):
    if not ENABLED:
        return
    cls = model_module.CenterGuidedPooling
    if getattr(cls, "_drse_tuned", False):
        return
    orig_init = cls.__init__

    def __init__(self, channels, center_size=3, attention_temperature=0.12, mode="guided"):
        orig_init(self, channels, center_size=CENTER_SIZE,
                  attention_temperature=TEMP, mode=mode)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor(INIT, dtype=torch.float32).log())

    __init__.__name__ = "init_drse_tuned"
    cls.__init__ = __init__
    cls._drse_tuned = True
    print(f"[drse] tuned: center_size={CENTER_SIZE} temp={TEMP} init={INIT}", flush=True)
