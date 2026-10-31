"""Dependency-light checks for syntax, configuration, and anonymous packaging."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(), filename=path.name)

    def test_no_deployment_metadata(self):
        for folder in ('verl', 'search_r1', 'configs', 'scripts'):
            for path in (ROOT / folder).rglob('*'):
                if path.suffix not in ('.py', '.json', '.yaml'):
                    continue
                text = path.read_text()
                self.assertIsNone(re.search(r'/(?:home|mnt)/[^\s]+', text), path.name)
                self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text), path.name)
                self.assertIsNone(re.search(r'\b[a-f0-9]{64}\b', text), path.name)

    def test_training_geometry(self):
        common = json.loads((ROOT / 'configs/common.json').read_text())
        for stage in (1, 2):
            c = common | json.loads((ROOT / f'configs/stage{stage}.json').read_text())
            self.assertEqual(c['actor_rollout_ref.rollout.n'], c['algorithm.cf_psr.expected_group_size'])
            dp = 8 // c['actor_rollout_ref.actor.ulysses_sequence_parallel_size']
            self.assertEqual(c['actor_rollout_ref.actor.ppo_mini_batch_size'] % dp, 0)
            self.assertEqual((c['actor_rollout_ref.actor.ppo_mini_batch_size'] // dp)
                             % c['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'], 0)

    def test_dry_run(self):
        for stage in ('1', '2'):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                '--stage', stage, '--runtime', 'RUNTIME_ROOT', '--model', 'MODEL_PATH',
                '--train', 'TRAIN_FILE', '--validation', 'VALIDATION_FILE',
                '--retrieval-endpoint', 'RETRIEVAL_ENDPOINT', '--output', 'OUTPUT_DIR', '--dry-run'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('verl.trainer.main_ppo', result.stdout)
            self.assertIn('cf_psr.enabled=true', result.stdout)


if __name__ == '__main__':
    unittest.main()
