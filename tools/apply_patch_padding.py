from pathlib import Path

# ---- data.py: PatchDataset padding mode -------------------------------------
data_path = Path("core/data.py")
text = data_path.read_text(encoding="utf8")
old_sig = (
    "        labels: np.ndarray | None = None,\n"
    "        augment: bool = False,\n"
    "    ) -> None:"
)
new_sig = (
    "        labels: np.ndarray | None = None,\n"
    "        augment: bool = False,\n"
    '        padding: str = "symmetric",\n'
    "    ) -> None:"
)
assert old_sig in text, "PatchDataset signature anchor missing"
text = text.replace(old_sig, new_sig, 1)
old_pad = '            mode="symmetric",\n        )'
new_pad = "            mode=str(padding),\n        )"
assert old_pad in text, "padding anchor missing"
text = text.replace(old_pad, new_pad, 1)
data_path.write_text(text, encoding="utf8")

# ---- trainer.py: CLI flag + pass padding to every PatchDataset --------------
trainer_path = Path("core/trainer.py")
text = trainer_path.read_text(encoding="utf8")

anchor = (
    '    parser.add_argument(\n'
    '        "--band_subset",\n'
)
assert anchor in text, "band_subset anchor missing"
flag = (
    '    parser.add_argument(\n'
    '        "--patch_padding",\n'
    '        type=str,\n'
    '        default="symmetric",\n'
    '        choices=["symmetric", "constant"],\n'
    '        help="Patch padding: symmetric (ours) or constant zeros (official MSDA).",\n'
    '    )\n'
)
text = text.replace(anchor, flag + anchor, 1)

# local alias inside build_loaders / evaluation loader
loader_anchor = "    source_train_dataset = PatchDataset("
assert loader_anchor in text, "build_loaders anchor missing"
alias = (
    '    patch_padding = str(getattr(args, "patch_padding", "symmetric") or "symmetric")\n'
    + loader_anchor
)
text = text.replace(loader_anchor, alias, 1)

eval_anchor = "    dataset = PatchDataset(\n        target_image,"
assert eval_anchor in text, "eval loader anchor missing"
eval_alias = (
    '    patch_padding = str(getattr(args, "patch_padding", "symmetric") or "symmetric")\n'
    + eval_anchor
)
text = text.replace(eval_anchor, eval_alias, 1)

pairs = [
    (
        'labels=source_train_labels,\n        augment=bool(getattr(args, "source_augment", True)),\n    )',
        'labels=source_train_labels,\n        augment=bool(getattr(args, "source_augment", True)),\n        padding=patch_padding,\n    )',
    ),
    (
        "labels=source_validation_labels,\n            augment=False,\n        )",
        "labels=source_validation_labels,\n            augment=False,\n            padding=patch_padding,\n        )",
    ),
    (
        "labels=None,\n        augment=False,\n    )",
        "labels=None,\n        augment=False,\n        padding=patch_padding,\n    )",
    ),
    (
        "labels=labels,\n        augment=False,\n    )",
        "labels=labels,\n        augment=False,\n        padding=patch_padding,\n    )",
    ),
]
for old, new in pairs:
    assert old in text, f"anchor missing: {old[:40]!r}"
    text = text.replace(old, new, 1)
trainer_path.write_text(text, encoding="utf8")
print("patched data.py and trainer.py")

# ---- configs: official zero padding -----------------------------------------
for dataset in ("pavia", "hyrank", "houston"):
    config = Path(f"datasets/{dataset}/config_msda.yaml")
    body = config.read_text(encoding="utf8")
    if "patch_padding" not in body:
        body = body.replace(
            "  input_normalization:",
            "  patch_padding: constant\n  input_normalization:",
            1,
        )
    config.write_text(body, encoding="utf8")

import yaml  # noqa: E402

for dataset in ("pavia", "hyrank", "houston"):
    conf = yaml.safe_load(Path(f"datasets/{dataset}/config_msda.yaml").read_text(encoding="utf8"))["training"]
    print(dataset, "| padding", conf.get("patch_padding"), "| patch", conf["patch_size"], "| norm", conf["input_normalization"])
