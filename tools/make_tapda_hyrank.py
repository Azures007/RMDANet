"""Generate TAPDA's HyRANK entry script (Dioni -> Loukia) from up2pc.py.

The TAPDA release ships three entry scripts (Pavia / Houston / YRD) but no
HyRANK script, although its ``util/utils.py`` already provides
``load_data_hyrank``.  This generator copies ``up2pc.py`` and rewrites:

    utils.load_data_pavia(...)          -> utils.load_data_hyrank(...)
    .../Pavia/paviaU.mat                -> datasets/HyRANK/Dioni.mat
    .../Pavia/paviaU_gt_7.mat           -> datasets/HyRANK/Dioni_gt_out68.mat
    .../Pavia/pavia.mat                 -> datasets/HyRANK/Loukia.mat
    .../Pavia/pavia_gt_7.mat            -> datasets/HyRANK/Loukia_gt_out68.mat
    NUM_CLASSES = 7 / num_classes = 7   -> 12
    N_BANDS = 102 / n_bands = 102       -> 176
    --halfwidth default 4               -> default 1   (patch 3, as in MSDA)
    run name UP_to_PC                   -> Dioni_to_Loukia

The generated file is validated BEFORE it is written, so a failed check never
leaves a broken script behind.  Idempotent (always regenerated from up2pc.py).

Usage:
    python tools/make_tapda_hyrank.py --tapda-dir /root/autodl-tmp/tapda
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MAT_MAP = {
    "paviaU.mat": "datasets/HyRANK/Dioni.mat",
    "paviaU_gt_7.mat": "datasets/HyRANK/Dioni_gt_out68.mat",
    "pavia.mat": "datasets/HyRANK/Loukia.mat",
    "pavia_gt_7.mat": "datasets/HyRANK/Loukia_gt_out68.mat",
}
MAT_PATTERN = re.compile(
    r"['\"](?P<path>[^'\"]*(?:paviaU_gt_7|paviaU|pavia_gt_7|pavia)\.mat)['\"]"
)


def rewrite(text: str) -> str:
    text = text.replace("utils.load_data_pavia(", "utils.load_data_hyrank(")
    text = MAT_PATTERN.sub(
        lambda match: repr(MAT_MAP[match.group("path").split("/")[-1]]), text
    )
    text = re.sub(r"\b(num_classes|NUM_CLASSES)\s*=\s*7\b", r"\g<1> = 12", text)
    text = re.sub(r"\b(n_bands|N_BANDS)\s*=\s*102\b", r"\g<1> = 176", text)
    text = re.sub(r"('--halfwidth',\s*default=)4\b", r"\g<1>1", text)
    text = text.replace("UP_to_PC", "Dioni_to_Loukia")
    text = text.replace("UP→PC", "Dioni→Loukia")
    text = text.replace("Pavia University → Pavia Center", "Dioni → Loukia")
    return text


def verify(generated: str) -> dict:
    checks = {
        "load_data_hyrank calls": generated.count("utils.load_data_hyrank("),
        "HyRANK paths": generated.count("datasets/HyRANK/"),
        "class count rewritten": len(
            re.findall(r"\b(?:num_classes|NUM_CLASSES)\s*=\s*12\b", generated)
        ),
        "band count rewritten": len(
            re.findall(r"\b(?:n_bands|N_BANDS)\s*=\s*176\b", generated)
        ),
        "halfwidth default 1": len(re.findall(r"'--halfwidth',\s*default=1\b", generated)),
        "leftover pavia paths": len(re.findall(r"Pavia/", generated)),
        "leftover class 7": len(re.findall(r"\b(?:num_classes|NUM_CLASSES)\s*=\s*7\b", generated)),
        "leftover bands 102": len(re.findall(r"\b(?:n_bands|N_BANDS)\s*=\s*102\b", generated)),
    }
    problems = []
    if checks["load_data_hyrank calls"] < 2:
        problems.append("expected two load_data_hyrank calls")
    if checks["HyRANK paths"] != 4:
        problems.append("expected four HyRANK .mat paths")
    if checks["class count rewritten"] < 1:
        problems.append("class count (NUM_CLASSES) not rewritten")
    if checks["band count rewritten"] < 1:
        problems.append("band count (N_BANDS) not rewritten")
    if checks["leftover pavia paths"]:
        problems.append("Pavia paths still present")
    if checks["leftover class 7"]:
        problems.append("NUM_CLASSES still 7")
    if checks["leftover bands 102"]:
        problems.append("N_BANDS still 102")
    if problems:
        raise RuntimeError("; ".join(problems))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapda-dir", default="/root/autodl-tmp/tapda")
    parser.add_argument("--source", default="up2pc.py")
    parser.add_argument("--target", default="dioni2loukia.py")
    options = parser.parse_args()

    root = Path(options.tapda_dir)
    source = root / options.source
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf8")
    if "load_data_pavia" not in text:
        raise RuntimeError("up2pc.py does not call utils.load_data_pavia; inspect manually")

    generated = rewrite(text)
    checks = verify(generated)

    target = root / options.target
    target.write_text(generated, encoding="utf8")
    print("generated:", target)
    for name, count in checks.items():
        print(f"  {name}: {count}")
    print("OK")
    print("next:")
    print(f"  python tools/patch_tapda_single_seed.py --tapda-dir {root} --script {options.target} --num-seeds 1")
    print(f"  python tools/patch_tapda_reliab.py --tapda-dir {root} --script {options.target}")
    print(f"  cd {root} && python -u {options.target} --gpu 0")


if __name__ == "__main__":
    main()
