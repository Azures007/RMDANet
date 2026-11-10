#!/usr/bin/env python3
"""Print OA / AA / Kappa and per-class accuracy from target_final_once.json."""
import json
import sys
from pathlib import Path


def report(path: Path) -> None:
    if not path.exists():
        print(f'[MISSING] {path}')
        return
    data = json.loads(path.read_text(encoding='utf8'))
    metrics = data['metrics']
    print('=' * 64)
    print(f'FILE : {path}')
    print(f"OA    : {metrics['OA']:.2f} %")
    print(f"AA    : {metrics['AA']:.2f} %")
    print(f"Kappa : {metrics['Kappa']:.4f}")
    class_accuracy = metrics.get('class_accuracy') or []
    print('PER-CLASS ACCURACY:')
    for index, accuracy in enumerate(class_accuracy, start=1):
        print(f'  class {index:2d} : {accuracy:6.2f} %')
    matrix = metrics.get('confusion_matrix')
    if matrix:
        total = sum(sum(row) for row in matrix)
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        print(f'  pixels={total} correct={correct} check_OA={100.0 * correct / max(1, total):.2f} %')
    selection = data.get('selection') or {}
    if selection:
        print(f"selected_epoch : {selection.get('main_best_epoch')} "
              f"(source-val {selection.get('source_validation_accuracy')})")
    print('=' * 64)


def main(argv):
    if len(argv) < 2:
        print('usage: python scripts/report_metrics.py <target_final_once.json> [...]')
        return 2
    for item in argv[1:]:
        report(Path(item))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
