"""Local output directories and stream logging for DCL experiments."""

import json
import os
import sys
from datetime import datetime

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

def safe_dir_name(name):
    invalid_chars = '<>:"/\\|?*'
    cleaned = ''.join('_' if ch in invalid_chars else ch for ch in str(name).strip())
    cleaned = '_'.join(cleaned.split())
    return cleaned or 'default'

def create_experiment_dir(args):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    dataset_dir = safe_dir_name(args.dataset_name)
    if args.run_name:
        run_name = safe_dir_name(args.run_name)
    else:
        run_name = safe_dir_name(f'{args.log_name}_{timestamp}')

    base_dir = os.path.join(args.output_root, dataset_dir, run_name)
    output_dir = base_dir
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = f'{base_dir}_{suffix}'
        suffix += 1
    os.makedirs(output_dir, exist_ok=False)
    return output_dir

def save_run_metadata(args, output_dir):
    args.output_dir = output_dir
    with open(os.path.join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(output_dir, 'command.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join([sys.executable] + sys.argv) + '\n')
