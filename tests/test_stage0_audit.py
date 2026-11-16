"""Stage-0 audit tests: entry-point consistency and effective-config logging."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import ablate_4rows  # noqa: E402
from trainer import audit_runtime_config, effective_config_snapshot  # noqa: E402


def test_full_row_matches_released_configuration():
    """`--row full` must be the released model: bidirectional fusion + spectral aux."""
    full = ablate_4rows.ROWS["full"]
    assert full["prat"] == 1
    assert full["rtml"] == 1
    assert full["fusion"] == "bidirectional"


def _namespace(**overrides):
    base = dict(
        ablation_id="full", run_name="unit_test", source_name="paviaU",
        target_name="paviaC", seed=1237, patch_size=7,
        spectral_fusion_mode="bidirectional", use_spectral_branch=True,
        use_spectral_aux=True, use_prat=True, pseudo_mode="history",
        use_probability_history=True, pooling_mode="guided", pool_center_size=5,
        pool_alpha_logit=-4.0, epochs=50, steps_per_epoch=40, batch_size=128,
        source_samples_per_class=180, source_train_ratio=0.38,
        source_budget_train_fraction=1.0, use_guard=True,
        use_confusion_loss=True, use_time_score=False, use_feature_score=False,
        use_spatial_reliability=True, use_ema_teacher=True,
        pseudo_min_per_class=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_effective_snapshot_records_key_fields(monkeypatch):
    monkeypatch.setenv("ABL_POOL_CENTER_SIZE", "5")
    snapshot = effective_config_snapshot(_namespace())
    assert snapshot["spectral_fusion_mode"] == "bidirectional"
    assert snapshot["use_spectral_aux"] is True
    assert snapshot["patch_size"] == 7
    assert snapshot["pool_alpha_logit"] == -4.0
    assert "lambda_cca" in snapshot["loss_lambdas"]
    assert snapshot["env"]["ABL_POOL_CENTER_SIZE"] == "5"


def test_audit_flags_noop_switches_and_pseudo_min_per_class(tmp_path):
    snapshot = audit_runtime_config(_namespace(), tmp_path)
    text = " ".join(snapshot["audit_warnings"])
    assert "use_guard" in text
    assert "use_confusion_loss" in text
    assert "pseudo_min_per_class=0" in text
    written = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert written["spectral_fusion_mode"] == "bidirectional"
    assert "audit_warnings" in written
