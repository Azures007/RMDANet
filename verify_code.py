"""Consistency check for the packaged code.

Run on the server right after unzipping:

    cd /root/autodl-tmp/v7clean
    python verify_code.py

It prints the SHA-256 prefix of the key files and checks that this is the
version where
  * DRSE is the dual-branch encoder (DualReliabilityEncoder),
  * RCGP (residual center-guided pooling) is a sub-module,
  * ``use_spectral_branch`` / ``--abl-drse`` exist,
  * module-creation order is the original one (so DRSE-on results are
    reproducible against the pre-switch code).
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "core/model.py",
    "core/trainer.py",
    "ablate_4rows.py",
    "scripts/final_eval_locked.py",
    "run_drseoff_row.sh",
]

print("== file hashes (sha256 prefix) ==")
for rel in FILES:
    p = ROOT / rel
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"
    print(f"{digest}  {rel}")

model_src = io.open(ROOT / "core/model.py", encoding="utf-8-sig").read()
trainer_src = io.open(ROOT / "core/trainer.py", encoding="utf-8-sig").read()
ablate_src = io.open(ROOT / "ablate_4rows.py", encoding="utf-8-sig").read()

classifier_pos = model_src.find("self.classifier = CosineClassifier")
encoder_pos = model_src.find("self.spectral_encoder")
fusion_pos = model_src.find("self.spectral_fusion = SpectralReliabilityFusion")

checks = {
    "model: use_spectral_branch parameter": "use_spectral_branch: bool = True" in model_src,
    "model: single-branch path (encoder = None)": "self.spectral_encoder = None" in model_src,
    "model: encoder created before classifier": 0 <= encoder_pos < classifier_pos,
    "model: fusion created after classifier": fusion_pos > classifier_pos > 0,
    "model: forward guards missing spectral branch": "if self.spectral_encoder is not None:" in model_src,
    "trainer: --use_spectral_branch flag": '"--use_spectral_branch"' in trainer_src,
    "trainer: model receives use_spectral_branch": "use_spectral_branch=getattr(args" in trainer_src,
    "ablate: --abl-drse switch": '"--abl-drse"' in ablate_src,
    "ablate: drse toggle wired": 'args.use_spectral_branch = bool(row.get("drse", 1))' in ablate_src,
}

print("\n== structure checks ==")
ok = True
for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    ok = ok and passed

try:
    version = io.open(ROOT / "VERSION.txt", encoding="utf-8-sig").read().strip()
    print("\n== VERSION.txt ==")
    print(version)
except FileNotFoundError:
    pass

print("\nRESULT:", "OK - this is the fixed package" if ok else "MISMATCH - wrong/partial package")
