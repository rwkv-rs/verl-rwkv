# RWKV MaxRL configuration

Run the canonical experiment directly from this checkout:

```bash
python -m verl.trainer.maxrl \
  --config examples/rwkv_trainer/config/maxrl_dapo_math_17k.toml
```

The canonical validation paths are materialized by
`examples/data_preprocess/rwkv_maxrl_math_eval.py`. This data conversion belongs
to the MaxRL recipe; it is not a benchmark evaluator. Full benchmark scoring
remains delegated to Helicopter's public LightEval command.

`verl.trainer.maxrl` owns the complete MaxRL contract. Launchers may forward the
TOML and explicit `--override` values, but they must not compile a second Hydra
configuration.

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
- `[[data.validation.suites]]` lists the validation parquets. Their native
  dataset fields remain untouched.
- `[algorithm]` exposes the global optimizer-step shape and objective
  coefficients. `prompts_per_step = 32` and `responses_per_prompt = 16` mean
  that one accepted global step contains 32 mixed-outcome groups and 512
  responses.
- `[reward]` selects the existing Verl reward manager and scorer.
- `[optimizer]` contains only optimizer semantics.
- `[generation.train]` and `[generation.validation]` contain the two fixed
  sampling contracts. Validation always samples; an unsupported `greedy`
  switch is not exposed.
- `[execution]` selects topology, precision, and RWKV state-passing.
  `[execution.rollout]` contains serving capacity and weight-transfer
  parameters, not optimizer batch semantics.
- `[evaluation]`, `[checkpoint]`, and `[logging]` contain lifecycle settings.
  Training-time validation uses Verl's validation loop. Full benchmark
  evaluation is a separate public LightEval command owned by Helicopter.

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
- validation `temperature = 0.96`, `top_p = 0.76`, `top_k = 32`,
  `presence_penalty = 1.0`, `frequency_penalty = 0.1`, and
  `penalty_decay = 0.988`.

Final Hydra overrides are limited to an explicit operational allowlist (for
example checkpoint paths, resume mode, logging, validation cadence, and rollout
capacity). Dataset, model, decoding, objective, and batch-shape fields cannot be
replaced from the command line.
