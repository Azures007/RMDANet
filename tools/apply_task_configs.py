"""Align the HyRANK / Houston MSDA configs with the official task settings.

Official MSDA scripts:
  dioni2loukia.py : patch = 2*1+1 = 3, N_BANDS = 176, backbone SSSE_test
  houston13218.py : patch = 2*3+1 = 7, N_BANDS = 48,  backbone SSSE_Houston
"""
from pathlib import Path

import yaml


def set_keys(path: str, values: dict) -> None:
    file = Path(path)
    lines = file.read_text(encoding="utf8").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    insert_at: int | None = None
    for line in lines:
        stripped = line.strip()
        key = stripped.split(":", 1)[0].strip() if ":" in stripped else ""
        if key in values:
            out.append(f"  {key}: {values[key]}")
            seen.add(key)
        else:
            out.append(line)
        if insert_at is None and stripped.startswith("input_normalization:"):
            insert_at = len(out)
    missing = [(key, value) for key, value in values.items() if key not in seen]
    if missing:
        position = len(out) if insert_at is None else insert_at
        for offset, (key, value) in enumerate(missing):
            out.insert(position + offset, f"  {key}: {value}")
    file.write_text("\n".join(out) + "\n", encoding="utf8")


set_keys("datasets/hyrank/config_msda.yaml", {"patch_size": 3, "backbone_class": "SSSE_test"})
set_keys(
    "datasets/houston/config_msda.yaml",
    {"patch_size": 7, "backbone_class": "SSSE_Houston", "band_subset": "'first:48'"},
)

for dataset in ("pavia", "hyrank", "houston"):
    config = yaml.safe_load(
        Path(f"datasets/{dataset}/config_msda.yaml").read_text(encoding="utf8")
    )["training"]
    print(
        dataset,
        "| patch", config["patch_size"],
        "| backbone", config.get("backbone_class", "SSSE_test"),
        "| band_subset", config.get("band_subset", ""),
    )
