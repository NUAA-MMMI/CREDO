"""Compose the source overlay with a compatible external framework package."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def patch_agent_dispatch(path):
    """Preserve registered constructor settings and dispatch uneven batches."""
    text = path.read_text()
    old = '_agent_loop_registry[agent_name] = {"_target_": fqdn}'
    new = '_agent_loop_registry.setdefault(agent_name, {"_target_": fqdn})'
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError('Unsupported agent registry implementation')
    old_split = '        chunkes = prompts.chunk(len(self.agent_loop_workers))'
    old_zip = 'zip(self.agent_loop_workers, chunkes, strict=True)'
    if old_split in text:
        if text.count(old_split) != 1 or text.count(old_zip) != 1:
            raise RuntimeError('Unsupported agent dispatch implementation')
        new_split = '''        worker_count = min(len(self.agent_loop_workers), len(prompts))
        if worker_count <= 0:
            raise ValueError("agent dispatch requires workers and prompts")
        base, remainder = divmod(len(prompts), worker_count)
        chunkes = []
        start = 0
        for index in range(worker_count):
            size = base + (1 if index < remainder else 0)
            chunkes.append(prompts[start : start + size])
            start += size'''
        text = text.replace(old_split, new_split)
        text = text.replace(old_zip, 'zip(self.agent_loop_workers[:worker_count], chunkes, strict=True)')
    elif ('worker_count = min(len(self.agent_loop_workers), len(prompts))' not in text
          or 'zip(self.agent_loop_workers[:worker_count], chunkes, strict=True)' not in text):
        raise RuntimeError('Unsupported agent dispatch implementation')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-verl', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, dest = args.base_verl.resolve(), args.output.resolve()
    if dest.exists() or source == dest or source in dest.parents:
        parser.error('output must be a new directory outside the framework package')
    version = source / 'version/version'
    if not version.is_file() or version.read_text().strip() != '0.7.1':
        parser.error('a compatible VERL 0.7.1 source package is required')
    required = ('trainer/main_ppo.py', 'experimental/agent_loop/agent_loop.py',
                'workers/rollout/replica.py', 'trainer/config/ppo_trainer.yaml')
    for name in required:
        if not (source / name).is_file():
            parser.error(f'missing framework component: {name}')
    shutil.copytree(source, dest / 'verl', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for path in (ROOT / 'verl').rglob('*.py'):
        target = dest / 'verl' / path.relative_to(ROOT / 'verl')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copytree(ROOT / 'search_r1', dest / 'search_r1',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch_agent_dispatch(dest / 'verl/experimental/agent_loop/agent_loop.py')
    print('Isolated CREDO runtime prepared. No training has been started.')


if __name__ == '__main__':
    main()
