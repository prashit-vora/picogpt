"""Prepare, submit, inspect, and download a private Kaggle LoRA training run."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'kaggle-job'
KERNEL = 'kezmanutz/picogpt-gpt2-lora'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'


def prepare(args):
    source = (ROOT / 'main.py').read_text()
    tree = ast.parse(source)
    # Read settings for reporting only; never override main.py.
    setting_names = {'batch_size', 'block_size', 'max_iters', 'eval_iters',
                     'eval_intervals', 'learning_rate'}
    settings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in setting_names:
                settings[target.id] = ast.literal_eval(node.value)

    # Only adapt asset paths to Kaggle's filesystem.
    class Configure(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if (isinstance(node.left, ast.Name) and node.left.id == 'MODEL_DIR'
                    and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant)
                    and node.right.value in ('tokenizer.json', 'model.safetensors')):
                return ast.copy_location(ast.Constant('/kaggle/temp/gpt2/' + node.right.value), node)
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                if node.value.endswith('/tokenizer.json'):
                    return ast.copy_location(ast.Constant('/kaggle/temp/gpt2/tokenizer.json'), node)
                if node.value.endswith('/model.safetensors'):
                    return ast.copy_location(ast.Constant('/kaggle/temp/gpt2/model.safetensors'), node)
            return node

    submitted = ast.unparse(ast.fix_missing_locations(Configure().visit(tree)))
    compile(submitted, 'submitted_main.py', 'exec')
    config = {**settings,
              'base_model': 'openai-community/gpt2', 'revision': REVISION,
              'rank': 8, 'alpha': 8}
    runner = '''import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.chdir('/kaggle/working')
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()
log = open('training.log', 'w', buffering=1)
sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'datasets', 'tokenizers', 'safetensors', 'huggingface_hub'], check=True)
import torch
from huggingface_hub import snapshot_download
if not torch.cuda.is_available():
    raise RuntimeError('This job requires a Kaggle GPU allocation.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
'''
    runner += f'config = {config!r}\n'
    runner += '''snapshot_download(config['base_model'], revision=config['revision'],
    local_dir='/kaggle/temp/gpt2', allow_patterns=['model.safetensors', 'tokenizer.json'])
Path('run_config.json').write_text(json.dumps(config, indent=2))
'''
    runner += f'comparison_source = {(ROOT / "prompt.py").read_text()!r}\n'
    runner += f'source = {submitted!r}\n'
    runner += '''Path('submitted_main.py').write_text(source)
namespace = {'__name__': '__main__', '__file__': '/kaggle/working/submitted_main.py'}
started = time.perf_counter()
completed = False
try:
    exec(compile(source, 'submitted_main.py', 'exec'), namespace)
    completed = True
    model = namespace['m']
    model.eval()
    comparison = {'__name__': 'comparison_helpers'}
    exec(compile(comparison_source, 'prompt.py', 'exec'), comparison)
    adapters = [layer for layer in model.modules() if isinstance(layer, namespace['LoRALinear'])]
    for prompt in ['Once upon a time, a little rabbit', 'One day, Lily found a']:
        idx = torch.tensor([namespace['encode'](prompt)], device=namespace['device'])
        comparison['compare'](model, adapters, idx, namespace['decode'], 80)
finally:
    model = namespace.get('m')
    if model is not None:
        adapters = {name: p.detach().cpu() for name, p in model.named_parameters()
                    if name.endswith(('.A.weight', '.B.weight'))}
        if adapters:
            torch.save({'adapters': adapters, 'config': config,
                        'training_completed': completed}, 'lora_adapters.pt')
    metrics = {'training_completed': completed,
               'elapsed_seconds_including_data_and_evaluation': time.perf_counter() - started,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
    Path('run_summary.json').write_text(json.dumps(metrics, indent=2))
    print(metrics)
'''
    compile(runner, 'train.py', 'exec')
    JOB.mkdir(exist_ok=True)
    (JOB / 'train.py').write_text(runner)
    metadata = dict(id=KERNEL, title='picogpt GPT2 LoRA', code_file='train.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=True,
                    dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    (JOB / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Prepared private Kaggle job:', KERNEL)
    print(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'push', 'status', 'download'])
    parser.add_argument('--accelerator', help='Optional Kaggle accelerator identifier')
    args = parser.parse_args()
    if args.command in ('prepare', 'push'):
        prepare(args)
    if args.command == 'push':
        command = ['kaggle', 'kernels', 'push', '-p', str(JOB)]
        if args.accelerator:
            command += ['--accelerator', args.accelerator]
        subprocess.run(command, check=True)
    elif args.command == 'status':
        subprocess.run(['kaggle', 'kernels', 'status', KERNEL], check=True)
    elif args.command == 'download':
        destination = ROOT / 'kaggle-results' / datetime.now().strftime('%Y%m%d-%H%M%S')
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['kaggle', 'kernels', 'output', KERNEL, '-p', str(destination)], check=True)
        print('Saved results to', destination)


if __name__ == '__main__':
    main()
