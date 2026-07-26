# RWKV MaxRL configuration

Run the canonical experiment directly from this checkout:

```bash
python -m verl.trainer.maxrl \
  --config examples/rwkv_trainer/config/maxrl_dapo_math_17k.toml
```

`verl.trainer.maxrl` owns the complete MaxRL contract. Launchers may forward the
TOML and explicit `--override` values, but they must not compile a second Hydra
configuration.

MaxRL does not implement a second benchmark evaluator or install LightEval.
Its `math_verify` scorer is only the training reward used to filter effective
groups. The `[evaluation]` lifecycle calls Helicopter's public LightEval command
before training and at the configured optimizer-step interval:

```bash
helicopter eval --config configs/eval/maxrl_math.toml --env-file .env.remote
```

The trainer first writes the current actor to
`global_step_<N>/actor/rwkv_lm.pth`, sleeps its rollout replicas, and supplies
the relative checkpoint and result paths to the command through
`MAXRL_EVAL_WEIGHT` and `MAXRL_EVAL_RESULT_PATH`. The command must write a JSON
object with a non-empty numeric `metrics` map. Rollout replicas wake after the
command exits, including on failure.

## User-facing sections

- `[experiment]` names the run, selects the logging project and seed, and sets
  `candidate_dataset_passes`. A value of zero means no preset epoch limit, so
  the operator stops the run manually.
- `[model]` selects the checkpoint and RWKV prompt style. The model context is
  derived from the checkpoint filename's single `ctxN` suffix.
- `[data.train]` lists the already materialized parquet inputs and the
  conversation column. The DAPO recipe uses every unique row from
  `open-r1/DAPO-Math-17k-Processed`; it does not repeat or randomly subsample
  the parquet.
- `[algorithm]` exposes the global optimizer-step shape and objective
  coefficients. `prompts_per_step = 32` and `responses_per_prompt = 16` mean
  that one accepted global step contains 32 mixed-outcome groups and 512
  responses.
- `[reward]` selects the existing Verl reward manager and scorer.
- `[optimizer]` contains only optimizer semantics.
- `[generation.train]` contains the fixed training sampling contract.
- `[execution]` selects topology, precision, and RWKV state-passing.
  `[execution.rollout]` contains serving capacity and weight-transfer
  parameters, not optimizer batch semantics.
- `[evaluation]` contains only the original lifecycle trigger and the external
  command. The benchmark list and publication policy belong to the referenced
  Helicopter eval config.
- `[checkpoint]` sets the cadence and a directory below `WEIGHT_PATH`, so the
  LightEval command can load the native actor checkpoint directly.
- `[logging]` contains training logger settings.

## Derived invariants

The entry point derives or enforces these values instead of exposing redundant
switches:

- model context from checkpoint `ctxN`;
- per-request output budget as model context minus that request's templated
  prompt length;
- no dataset-wide prompt-length scan and no configurable prompt/output maxima;
- EOS stopping enabled unconditionally;
- one response per actor/ref/log-prob microbatch slot, with dynamic token
  microbatching disabled;
- one PPO epoch and one global mini-batch per optimizer step;
- synchronous V1 hybrid training with the native RWKV actor/ref engines;
- MaxRL effective-group filtering and refill until 32 mixed-outcome groups are
  available;
- training `top_p = 0.95`;
- no Verl validation scorer or LightEval Python dependency; validation metrics
  come back from the configured command.

Final Hydra overrides are limited to an explicit operational allowlist (for
example checkpoint paths, resume mode, logging, and rollout capacity). Dataset,
model, decoding, objective, batch-shape, and evaluation fields cannot be
replaced from the command line.
