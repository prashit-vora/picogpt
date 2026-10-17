"""Submit the latest saved main.py to Kaggle without editing it."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'kaggle-job'
KERNEL = 'kezmanutz/picogpt-addition'


def prepare():
    source = (ROOT / 'main.py').read_text()
    ast.parse(source)  # Catch syntax errors before uploading.
    JOB.mkdir(exist_ok=True)
    metadata = {
        'id': KERNEL, 'title': 'picogpt addition',
        'code_file': 'train.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'dataset_sources': [],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': [],
    }
    (JOB / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    runner = '''import builtins
import os
from pathlib import Path
import sys
import torch

os.chdir('/kaggle/working')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable: check the Kaggle GPU allocation.')

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()

log = open('training.log', 'w', buffering=1)
sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)
# The local script asks how many interactive equations to test after evaluation.
# Batch jobs cannot accept terminal input, so request zero interactive tests.
def batch_input(prompt=''):
    if 'how many' in prompt.lower():
        print(prompt + '0 [noninteractive Kaggle run]')
        return '0'
    raise RuntimeError('Unexpected interactive input in Kaggle job: ' + prompt)
builtins.input = batch_input
'''
    runner += '\nsource = ' + repr(source) + '\n'
    runner += '''Path('submitted_main.py').write_text(source)
namespace = {'__name__': '__main__', '__file__': str(Path('submitted_main.py').resolve())}
try:
    exec(compile(source, 'submitted_main.py', 'exec'), namespace)
finally:
    # Save even if evaluation fails after training; this is not automatic resume.
    model = namespace.get('m')
    if model is not None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {key: namespace.get(key) for key in (
                'n_embd', 'n_head', 'n_layer', 'block_size', 'dropout',
                'vocab_size', 'batch_size', 'learning_rate', 'max_iters')},
            'vocab': namespace.get('vocab'),
            'source': source,
            'last_loop_index': namespace.get('epoch'),
        }
        optimizer = namespace.get('optimizer')
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, 'checkpoint.pt')
        print('Saved checkpoint.pt, submitted_main.py, and training.log')
'''
    ast.parse(runner)
    (JOB / 'train.py').write_text(runner)
    print('Prepared latest saved main.py for', KERNEL, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'push', 'status', 'download'])
    args = parser.parse_args()
    if args.command in ('prepare', 'push'):
        prepare()
    if args.command == 'push':
        subprocess.run(['kaggle', 'kernels', 'push', '-p', str(JOB)], check=True)
    elif args.command == 'status':
        subprocess.run(['kaggle', 'kernels', 'status', KERNEL], check=True)
    elif args.command == 'download':
        from datetime import datetime
        destination = ROOT / 'kaggle-results' / datetime.now().strftime('%Y%m%d-%H%M%S')
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['kaggle', 'kernels', 'output', KERNEL, '-p', str(destination)], check=True)
        print('Downloaded outputs to', destination)


if __name__ == '__main__':
    main()
