"""Resource-agnostic launcher; never starts a job at import time."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--retrieval-endpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('npu', 'cuda'), default='npu')
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/common.json').read_text())
    config.update(json.loads((ROOT / f'configs/stage{args.stage}.json').read_text()))
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    config.update({
        'actor_rollout_ref.model.path': args.model,
        'data.train_files': args.train,
        'data.val_files': args.validation,
        'trainer.device': args.device,
        'trainer.n_gpus_per_node': args.devices,
        'trainer.experiment_name': f'credo_stage{args.stage}',
        'trainer.default_local_dir': str(output / 'checkpoints'),
        'trainer.validation_data_dir': str(output / 'validation'),
        'trainer.rollout_data_dir': str(output / 'rollouts'),
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(output / 'agent.yaml'),
        'custom_reward_function.path': str(runtime / 'search_r1/reward_fn_binary.py'),
        'custom_reward_function.name': 'compute_score',
    })
    for override in args.set:
        key, sep, value = override.partition('=')
        if not sep or not key:
            parser.error('--set requires KEY=VALUE')
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    sp = int(config['actor_rollout_ref.actor.ulysses_sequence_parallel_size'])
    world = int(config['trainer.n_gpus_per_node']) * int(config['trainer.nnodes'])
    mini = int(config['actor_rollout_ref.actor.ppo_mini_batch_size'])
    micro = int(config['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'])
    group = int(config['actor_rollout_ref.rollout.n'])
    if min(world, sp, mini, micro, group) < 1 or world % sp:
        parser.error('invalid parallel geometry')
    dp = world // sp
    if mini % dp or (mini // dp) % micro:
        parser.error('minibatch must divide DP and local minibatch must divide microbatch')
    if int(config['algorithm.cf_psr.expected_group_size']) != group:
        parser.error('CF group size must equal rollout.n')
    if int(config['data.train_batch_size']) * group % mini:
        parser.error('rollout batch must divide into complete PPO minibatches')
    command = [sys.executable, '-m', 'verl.trainer.main_ppo']
    command += [f'++{k}={json.dumps(v, ensure_ascii=False)}' for k, v in config.items()]
    if args.dry_run:
        print(shlex.join(command))
        return
    if not (runtime / 'verl/trainer/ppo/ocor_runtime.py').is_file():
        parser.error('prepare a CREDO runtime first')
    if output.exists():
        parser.error('output must be a new directory to avoid overwriting a run')
    if args.retrieval_endpoint == 'RETRIEVAL_ENDPOINT':
        parser.error('replace symbolic resource values before training')
    output.mkdir(parents=True)
    agent = json.loads((ROOT / 'configs/agent.json').read_text())
    if args.stage == 2:
        agent[0].update(topk=20, max_start_length=1500, max_obs_length=1500,
                        prompt_length=8192, response_length=8192,
                        max_tokens_per_turn=1024, max_model_len=9216)
    agent[0]['search_url'] = args.retrieval_endpoint
    agent[0]['prompt_length'] = config['data.max_prompt_length']
    agent[0]['response_length'] = config['data.max_response_length']
    agent[0]['max_model_len'] = config['actor_rollout_ref.rollout.max_model_len']
    # JSON is valid YAML and preserves quoting of externally supplied values.
    (output / 'agent.yaml').write_text(json.dumps(agent, indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(runtime) + os.pathsep + env.get('PYTHONPATH', '')
    env['CREDO_SEARCH_URL'] = args.retrieval_endpoint
    env['CREDO_REQUIRE_RETRIEVAL'] = '1'
    raise SystemExit(subprocess.call(command, cwd=runtime, env=env))


if __name__ == '__main__':
    main()
