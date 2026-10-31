# CREDO — Anonymous Training Code

This package contains the main CREDO training implementation and its integration
with the legacy VERL FSDP actor. It contains no model weights, datasets,
predictions, experiment records, host configuration, or checkpoint identifiers.
All resources are supplied by the user through symbolic configuration fields.

## Layout

```text
verl/
  trainer/ppo/             Local credit, outcome redistribution, GRPO and trainer
  workers/actor/           Clipped policy loss and gradient accumulation
  workers/config/         Method configuration fields
  workers/fsdp_workers.py  Distributed worker integration
  utils/config.py         Parallel-batch validation
search_r1/                Search loop and terminal-answer reward
configs/                  Resource-independent two-stage training recipes
scripts/                  Isolated runtime preparation and training entry points
tests/                    Synthetic method and packaging checks
```

The `verl` directory is a source overlay, not an independently installable copy
of the full framework. `prepare_runtime.py` combines it with a compatible VERL
0.7.1 package in a **new directory**. It does not modify the supplied framework,
site packages, or any existing experiment. Upstream framework code and accelerator
dependencies must be installed separately. The supplied worker path uses legacy
FSDP and retains NPU compatibility; accelerator execution requires a compatible
PyTorch, device extension, and vLLM installation.

## Method

1. Group trajectories by their original question identifier before distributed
   or microbatch splitting. Compute normalized answer reward and GRPO advantages.
2. Parse generated action spans; exclude tool observations and padding from
   trainable tokens. Compute local credit from query, evidence, and answer quality,
   using valid same-question comparisons and the implemented stage budgets.
3. For eligible trajectories, redistribute a fixed fraction of outcome-policy
   weight toward reasoning/answering tokens while preserving each trajectory's
   total weight. Terminal rewards and GRPO advantages are unchanged.
4. Optimize clipped outcome and local objectives; use the original actor mask
   for KL and entropy terms.

The preserved main implementation uses Rank-priority final selection: the last
Rank block takes priority over Filter blocks. This package does not silently
substitute a chronological-last selection rule or a new no-gating variant.
The local answer parser and the terminal reward parser remain separate; they
must not be described as an identical strict-answer criterion. The supplied
terminal reward is binary **normalized substring matching**, with format and
retrieval bonuses disabled. Local answer quality uses token F1.

Outcome loss is normalized within each static microbatch by the sum of outcome
weights (equal to its valid actor-token count), then scaled by inverse gradient
accumulation. FSDP and sequence-parallel scaling yield averaging over effective
data-parallel replicas. This is not a single global-token-denominator objective.
Local loss uses its separate minibatch weight normalization.

The search adapter retains raw observation insertion and the original trajectory
container truncation. It does not guarantee that observations or final answers
survive a tight budget. Generated-token limits and the container budget (which
also holds observations) are distinct configuration fields.

## Preparation

Install the required framework/backend dependencies in an isolated environment.
The requirements file lists Python-level dependencies; it does not select an
accelerator-specific binary distribution.

```bash
python scripts/prepare_runtime.py --base-verl BASE_VERL_PACKAGE --output RUNTIME_ROOT
```

`BASE_VERL_PACKAGE` denotes the framework's `verl` package directory, not its
parent repository. `RUNTIME_ROOT` must not exist. These are symbolic resource
names, not literal resource locations.

## Resource contract

Supply each training/validation split as a compatible parquet file. Each row
contains `prompt` (chat messages), `data_source` (an arbitrary task identifier),
`reward_model.ground_truth.target` (acceptable answer strings), and optional
`extra_info` metadata. Prompts and tokenizer chat templates are provided by the
caller; this package does not embed a benchmark prompt. The trainer assigns
same-question rollout identifiers before repetition.

The retrieval service receives `queries`, `topk`, and `return_scores`; its
response must follow the document structure consumed by
`search_r1/search_agent_loop.py`. Start the service separately and supply its
endpoint as `RETRIEVAL_ENDPOINT`. No service credentials or default address are
included.

## Training

```bash
python scripts/train.py --stage 1 --runtime RUNTIME_ROOT \
  --model MODEL_PATH --train TRAIN_FILE --validation VALIDATION_FILE \
  --retrieval-endpoint RETRIEVAL_ENDPOINT --output OUTPUT_DIR

python scripts/train.py --stage 2 --runtime RUNTIME_ROOT \
  --model STAGE1_HF_MODEL --train TRAIN_FILE --validation VALIDATION_FILE \
  --retrieval-endpoint RETRIEVAL_ENDPOINT --output OUTPUT_DIR
```

Use distinct output directories. Stage 2 loads the selected Stage 1 HF model
with a fresh optimizer; it does not automatically resume optimizer state.
No training is started merely by preparing this submission.
`--dry-run` prints the launch command without starting training.
`--device` and `--devices` select the accelerator configuration; use `--set
KEY=VALUE` for Hydra overrides. Keep PPO minibatches divisible by effective DP
size and local minibatches divisible by microbatch size. The provided recipes
encode one eight-device geometry, not a claim of hardware-independent throughput.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic inputs only. They check reward semantics, weight conservation,
evidence gating, token masks, local target scoring, and anonymous-package content.
They do not replace an accelerator-backed end-to-end training check.

Internal `ocor_*` identifiers are retained for source compatibility; the public
method name is CREDO. Diagnostic hashing used by deterministic control logic is
an algorithmic operation, not a recorded model/file fingerprint.

## Third-party code

Framework-derived files retain their third-party copyright notices. See
`LICENSE`, `LICENSE.third_party`, and `NOTICE` for attribution. These are upstream attributions, not
submission-author identities.
