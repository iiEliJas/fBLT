# fBLT Release TODO

Working document for the feature work needed before and after the 1.0 release.

Baseline: `9b7f5ac` (`fix(train): paper loss is now default`). Line numbers refer to that
tree and will drift as the items land. Re-check the surrounding function before editing.

Scope: this file covers the features identified in the nanoGPT comparison. It does not
repeat the release-engineering gates (versioning, docs path fixes, CI matrix) except in the
short checklist at the end.

Legend:

- **Before 1.0**: small, high-value items that make the trainer usable and observable.
- **1.x**: larger training/scaling work that should not block a 1.0 cut.
- **Effort**: rough estimate for one developer familiar with the codebase.
- **Risk**: what can go wrong or what needs re-validation.

---

## Priority summary

| # | Item | Phase | Effort | Risk |
|---|------|-------|--------|------|
| 1 | Best-checkpoint saving | Before 1.0 | < 1 day | Low |
| 2 | Metrics JSONL + optional W&B | Before 1.0 | 1-2 days | Low |
| 3 | LR warmup + cosine/min-lr | Before 1.0 | < 1 day | Low |
| 4 | Training telemetry (params, throughput, MFU) | Before 1.0 | 0.5-1 day | Low |
| 5 | `--num-samples` and `--top-k` | Before 1.0 | 0.5-1 day | Low |
| 6 | Profiling target + docs | Before 1.0 | 0.5-1 day | Low |
| 7 | Gradient accumulation / batch > 1 | 1.x | 3-7 days | Medium |
| 8 | BF16 mixed-precision training | 1.x | 1-3 weeks | High |
| 9 | Selective weight decay | 1.x | 0.5 day | Medium |
| 10 | Checkpoint metadata v2 | 1.x | 1-2 days | Medium |
| 11 | Multi-GPU data parallel | 1.x | 3-6 weeks | High |
| 12 | macOS support | 1.x | 2-4 days | Low-Medium |
| 13 | Finetune recipe (optional) | Polish | 0.5-2 days | Low |
| 14 | Packaging extras (optional) | Polish | 1 day | Low |
| 15 | Demo checkpoint (optional) | Polish | 0.5 day + training | Low |
| 16 | Standard eval harness (optional) | Polish | 3-5 days | Medium |

Recommended cut: after items 1-6, 9, 10, and the release gates. Items 7, 8, 11, and 12
should ship in 1.1/1.2 rather than delaying 1.0.

---

# Before 1.0

## 1. [DONE] Best-checkpoint saving

**Status:** not started  
**Effort:** < 1 day  
**Depends on:** an eval corpus (`--eval-corpus`) and a nonzero `--eval-every`

### Why

nanoGPT keeps `best_val_loss` and can save only when validation improves
(`always_save_checkpoint = False`). fBLT only has periodic and final saves, so a run that
degrades late overwrites nothing but also gives you no easy "keep the best" artifact. The
trainer already computes causal BPB on held-out data; wiring that metric to a save decision
is cheap.

### Current state

- `csrc/train_args.h` has `save_path`, `save_every`, `save_optim_path`, `eval_path`,
  `eval_every`, `eval_windows`.
- `csrc/train_blt_d.c:903` performs periodic saves with `blt_format_save_path`, which
  substitutes the first `%d` with the step.
- `csrc/train_blt_d.c:916-957` is the periodic eval block. It computes `bpb` and
  `masked_acc`, prints `EVAL`, then frees the corpus buffer.
- `csrc/train_blt_d.c:968-976` saves the final weights before the final eval.
- `csrc/train_blt_d.c:979-1039` is the final eval block and `RESULT` line.

### Implementation

1. Add fields to `args_t` in `csrc/train_args.h`:

   ```c
   const char *save_best_path;
   int best_metric;              // 0 = causal_bpb (min), 1 = masked_acc (max)
   const char *save_best_optim_path;
   ```

   Add them to `parse_args` defaults in `csrc/train_args.c` and to the `usage()` text.

2. Add flags:

   ```text
   --save-best PATH          write weights when the eval metric improves
   --best-metric bpb|masked_acc   metric to optimize (default bpb)
   --save-best-optim PATH    write AdamW state alongside the best weights
   ```

3. Validate at startup, before the training loop:

   ```c
   BLT_REQUIRE(a.save_best_path == NULL || a.eval_path != NULL,
               "--save-best requires --eval-corpus");
   BLT_REQUIRE(a.save_best_path == NULL || a.eval_every > 0,
               "--save-best requires --eval-every > 0");
   ```

4. Track state in `main`:

   ```c
   double best_metric_value = (a.best_metric == 0) ? INFINITY : -INFINITY;
   size_t best_step = 0;
   ```

5. Refactor the repeated eval loop into one helper so the periodic and final eval paths
   share the metric computation. The helper should return `bpb`, `masked_acc`, and the
   `hits/total` pair. The periodic block at 916 and the final block at 979 are currently
   near-duplicates; a helper also reduces the chance of the best-checkpoint logic diverging
   between the two.

6. After an eval produces `bpb` and `masked_acc`:

   ```c
   const double metric = (a.best_metric == 0) ? bpb : masked_acc;
   const int valid = (a.best_metric == 0) ? isfinite(bpb) : (total > 0);
   const int improved = valid && ((a.best_metric == 0) ? (metric < best_metric_value)
                                                        : (metric > best_metric_value));
   if (a.save_best_path && improved) {
       best_metric_value = metric;
       best_step = step + 1;
       char path_buf[4096];
       blt_format_save_path(path_buf, sizeof(path_buf), a.save_best_path, step + 1);
       blt_model_save(model, path_buf);
       if (a.save_best_optim_path && aw) {
           char opath_buf[4096];
           blt_format_save_path(opath_buf, sizeof(opath_buf), a.save_best_optim_path, step + 1);
           adamw_state_save(aw, step + 1, opath_buf);
       }
       printf("[BEST] step=%zu %s=%.6f -> %s\n", step + 1,
              a.best_metric == 0 ? "causal_bpb" : "masked_acc", metric, path_buf);
       fflush(stdout);
   }
   ```

7. In the final eval block, run the same comparison after the final `bpb`/`masked_acc` is
   known. The final weights save at 968 happens before this eval; that is fine, because the
   best checkpoint is a separate artifact. Do not overwrite the final `--save-weights` file
   with the best checkpoint.

8. Add the fields to `TrainConfig` in `fblt/_config.py` so the Python wrapper exposes them:

   ```python
   save_best: typing.Optional[str] = None
   best_metric: str = "bpb"
   save_best_optim: typing.Optional[str] = None
   ```

   Do not add `save_best` to `_DEFAULT_OUTPUTS` in `fblt/train.py`; best-checkpoint saving
   should stay opt-in. `_to_argv` already skips `None` optionals.

9. Update `docs/CLI_REFERENCE.md` under I/O and evaluation, and add a short README example.

### Acceptance

- With `--save-best runs/x/best.fblt --eval-every 50 --eval-corpus data/heldout.bin`, a run
  writes `best.fblt` only when BPB improves, and the log shows `[BEST]` with the metric.
- The final `--save-weights` artifact is still written exactly as before.
- `--save-best` without an eval corpus exits with a clear error before training starts.
- A NaN or non-finite BPB never produces a best checkpoint.
- `%d` in `--save-best` is replaced with the step, matching `--save-weights`.

### Tests and docs

- Add a CI smoke step that trains for a handful of steps with `--eval-every 1
  --save-best`, then asserts the file exists and is a valid `.fblt` checkpoint (load it with
  `infer` or a small C test).
- Add a unit test for the comparison helper if the metric logic is extracted. Cover
  `bpb` min, `masked_acc` max, NaN, and zero-denominator cases.
- Document that best-metric tracking does not persist across process restarts unless you
  also reload the best optimizer state or record the best value externally. If persistence
  is wanted, add `best_metric_value`/`best_step` to the optimizer metadata in item 10.

### Risks

- The final eval happens after the final weight save. Keep the two artifacts separate; do
  not "fix" this by moving the final save after eval without checking resume semantics.
- `masked_acc` can be undefined when the final window has no masked cells. Guard on
  `total > 0`.
- Frequent eval with `--eval-every` can dominate runtime on large eval corpora. This is
  existing behavior; the best-checkpoint feature does not change it.

---

## 2. [DONE] Metrics JSONL + optional W&B

**Status:** not started  
**Effort:** 1-2 days  
**Depends on:** nothing; item 1 benefits from the same event stream

### Why

nanoGPT logs to Weights & Biases when `wandb_log = True`, including train/val loss, LR, and
MFU. fBLT writes several human-readable log files (`--loss-log`, `--grad-norm-log`, etc.) and
prints events, but there is no single machine-readable stream and no experiment tracker.
That makes comparing runs, plotting curves, and resuming analysis harder than it needs to be.

### Current state

- `csrc/train_blt_d.c` prints:
  - per-report `step ... avg_loss ... ETA ...` at 889-899
  - `[TIMING]` at 836-842 when `BLT_TRAIN_TIMING` is set
  - `[MONITOR]` at 857-876
  - `EVAL step=...` at 953
  - `RESULT ...` at 1031
- `--loss-log`, `--grad-norm-log`, `--update-norm-log`, `--component-norm-log`,
  `--activation-dump-log`, `--batch-log` are opened in `csrc/train_blt_d.c:620-630`.
- `fblt/_runner.py` streams stdout/stderr line by line from the child process.
- `fblt/train.py` creates `runs/<name>/` and writes `resolved_config.yaml`.
- There is no `wandb`/`tensorboard`/`mlflow` reference anywhere in the repo.

### Implementation

#### 2a. C-side JSONL

1. Add `--metrics-log FILE` to `args_t` and `parse_args`.

2. Open it once near the other log files:

   ```c
   FILE *metrics_fp = NULL;
   if (a.metrics_log) {
       if (train_ensure_parent_dir(a.metrics_log) == 0) metrics_fp = fopen(a.metrics_log, "w");
       BLT_REQUIRE(metrics_fp != NULL, "train_blt_d: cannot open --metrics-log %s", a.metrics_log);
   }
   ```

3. Emit one JSON object per line. Keep the schema stable and numeric-only:

   ```json
   {"event":"train","step":1,"loss":5.123456,"lr":0.0001,"grad_norm":3.21,"step_ms":123.4,"bytes_per_s":389.1}
   {"event":"eval","step":100,"causal_bpb":2.4123,"masked_acc":0.3712,"best":true}
   {"event":"monitor","step":100,"l_clean":4.1,"clean_bpb":5.9,"mask_bpb":5.6,"train_acc":0.2,"novel_acc":0.18,"gap":0.02}
   {"event":"result","steps":1000,"causal_bpb":2.1012,"masked_acc":0.50,"params":2970000,"best_step":900}
   ```

   The C side can use `fprintf` directly; all values are numeric and the event names are
   fixed strings, so no general JSON writer is needed. Guard against `NaN`/`Inf` by writing
   `null` or skipping the event; most JSON parsers reject `NaN`.

4. Emit the `train` event after the optimizer step, where `lr`, `pre_clip_norm`, the step
   time, and the loss are all in scope. Move the LR computation (item 3) before the report
   so the logged LR is the one actually used.

5. Emit `eval` from the shared eval helper (item 1), including the `best` flag.

6. Emit `monitor` inside the existing monitor block.

7. Emit `result` immediately before cleanup at the end.

8. Close `metrics_fp` with the other file handles at 1044-1051.

#### 2b. Python W&B

1. Add `--wandb`, `--wandb-project`, `--wandb-run-name`, and `--wandb-entity` to
   `fblt/train.py`'s argparse. These are wrapper-only flags; do not add them to
   `TrainConfig` or `to_argv`.

2. Add `tracking = ["wandb"]` to `[project.optional-dependencies]` in `pyproject.toml`.

3. Default `--metrics-log` to `runs/<name>/metrics.jsonl` in `_DEFAULT_OUTPUTS` so the
   wrapper always has a stream to tail. Users who do not want the file can still override
   the path or run the C binary directly.

4. Add a tail thread. The simplest robust implementation is a small helper in
   `fblt/_runner.py`:

   ```python
   def tail_jsonl(path, on_event, stop_event, poll=0.25):
       # Open lazily: the C process may create the file after the child starts.
       while not stop_event.is_set() and not os.path.exists(path):
           time.sleep(poll)
       with open(path, "r") as fh:
           while not stop_event.is_set():
               line = fh.readline()
               if not line:
                   time.sleep(poll)
                   continue
               try:
                   on_event(json.loads(line))
               except json.JSONDecodeError:
                   # A partially flushed line can appear at the tail; retry next poll.
                   fh.seek(fh.tell() - len(line))
   ```

   The retry logic must not drop a line permanently. A simpler alternative is to read the
   whole file again from a byte offset on each poll.

5. In `fblt/train.py`, when `--wandb` is set:

   ```python
   import wandb

   run = wandb.init(
       project=args.wandb_project,
       name=args.wandb_run_name,
       entity=args.wandb_entity,
       config=resolved_config,
   )
   stop = threading.Event()
   t = threading.Thread(target=tail_jsonl, args=(metrics_path, run.log, stop))
   t.start()
   try:
       rc = run_binary(...)
   finally:
       stop.set()
       t.join(timeout=5)
       run.finish()
   ```

   `run.log` accepts a dict. Map `event` to a W&B section, for example
   `{"train/loss": ..., "train/lr": ...}` and `{"eval/bpb": ..., "eval/masked_acc": ...}`,
   so the dashboard is readable.

6. Do not require `wandb` when `--wandb` is absent. Import lazily and print a clear error
   with the install command if it is missing.

### Acceptance

- `--metrics-log runs/x/metrics.jsonl` produces one valid JSON object per line, parseable by
  `python -c "import json,sys; [json.loads(l) for l in sys.stdin]"`.
- The file contains at least `train`, `eval`, `monitor` (when enabled), and `result` events.
- `fblt-train --wandb --wandb-project fblt` streams live step and eval metrics.
- Training without either flag behaves exactly as before and creates no metrics file when
  the C binary is invoked directly.
- A killed process leaves a readable prefix of the file, not a corrupt one.

### Tests and docs

- Add a Python unit test for the JSONL tail helper using a temporary file and a fake
  callback.
- Add a CI smoke step that runs a tiny train with `--metrics-log`, parses every line, and
  asserts required keys.
- Document the schema in `docs/CLI_REFERENCE.md` and add a `docs/TRACKING.md` with a W&B
  example. Mention that metrics are local by default.

### Risks

- The C process may buffer stdout but writes the JSONL file with `fprintf`; add
  `setvbuf(metrics_fp, NULL, _IOLBF, 0)` or `fflush(metrics_fp)` after each event to keep
  the tail useful.
- Training and eval events can arrive out of order if eval runs inside the step. Emit them
  from the same thread; the C trainer is single-threaded.
- W&B may be unavailable in offline environments. The wrapper must fail gracefully and
  suggest `WANDB_MODE=offline`.

---

## 3. [DONE] LR warmup + cosine/min-lr

**Status:** not started  
**Effort:** < 1 day  
**Depends on:** nothing

### Why

nanoGPT uses linear warmup followed by cosine decay to `min_lr`:
`get_lr()` in `train.py:230-242`. fBLT only has step decay (`--lr-decay` and custom
`--lr-decay-steps`). Warmup matters for stability at larger batch sizes/LRs, and cosine is
the standard recipe for comparing against published training curves.

### Current state

- `csrc/train_blt_d.c:790-799` computes:

  ```c
  float lr = a.lr;
  if (a.lr_decay_count > 0) {
      for (int d = 0; d < a.lr_decay_count; d++)
          if (step >= a.lr_decay_steps[d]) lr *= a.lr_decay_factor;
  } else if (a.lr_decay) {
      if (step >= (end_step * 85) / 100) lr *= 0.09f;
      else if (step >= (end_step * 60) / 100) lr *= 0.3f;
  }
  ```

- `csrc/train_args.h` has `lr_decay`, `lr_decay_steps`, `lr_decay_count`,
  `lr_decay_factor`.
- The LR is not printed in the periodic report and is not logged.

### Implementation

1. Add fields to `args_t`:

   ```c
   size_t warmup_steps;
   int lr_schedule;      // 0 = current step decay, 1 = cosine
   float min_lr;         // < 0 = default to lr * 0.1
   int lr_schedule_given;
   ```

2. Add flags:

   ```text
   --warmup-steps N          linear warmup steps (default 0)
   --lr-schedule step|cosine  schedule after warmup (default step)
   --min-lr F                cosine floor (default lr * 0.1)
   ```

3. Reject conflicting schedules early:

   ```c
   BLT_REQUIRE(!(a.lr_schedule == 1 && (a.lr_decay || a.lr_decay_count > 0)),
               "--lr-schedule cosine cannot be combined with --lr-decay");
   ```

4. Add a pure helper, ideally in `csrc/train_eval.c`/`.h` or a new
   `csrc/train_schedule.c`/`.h`, so it is unit-testable without a model:

   ```c
   float train_lr_at_step(const args_t *a, size_t step, size_t end_step) {
       const float min_lr = (a->min_lr >= 0.0f) ? a->min_lr : 0.1f * a->lr;
       if (a->warmup_steps > 0 && step < a->warmup_steps) {
           return a->lr * (float)(step + 1) / (float)(a->warmup_steps + 1);
       }
       if (a->lr_schedule == 1) {
           const size_t start = a->warmup_steps;
           const size_t span = (end_step > start) ? (end_step - start) : 1;
           double progress = (step > start) ? (double)(step - start) / (double)span : 0.0;
           if (progress > 1.0) progress = 1.0;
           const double coeff = 0.5 * (1.0 + cos(M_PI * progress));
           return (float)(min_lr + coeff * (double)(a->lr - min_lr));
       }
       float lr = a->lr;
       if (a->lr_decay_count > 0) {
           for (int d = 0; d < a->lr_decay_count; d++)
               if (step >= a->lr_decay_steps[d]) lr *= a->lr_decay_factor;
       } else if (a->lr_decay) {
           if (step >= (end_step * 85) / 100) lr *= 0.09f;
           else if (step >= (end_step * 60) / 100) lr *= 0.3f;
       }
       return lr;
   }
   ```

   `M_PI` is not guaranteed by strict C99; use a local `const double pi =
   3.14159265358979323846;` or `acos(-1.0)`.

5. Replace the inline block in `train_blt_d.c` with
   `float lr = train_lr_at_step(&a, step, end_step);`.

6. Compute the LR before the periodic report and include it in the report line and the
   metrics event. This also makes item 4's telemetry accurate.

7. Resume semantics: use the absolute `step` (which starts at `--start-step`). Warmup is
   skipped automatically when `start_step >= warmup_steps`. Document this.

8. Add the fields to `TrainConfig` in `fblt/_config.py` and to `docs/CLI_REFERENCE.md`.

### Acceptance

- For `lr=1e-4`, `warmup_steps=10`, step 0 gives `1e-5`; step 9 gives `1e-4`; step 10
  starts decay.
- Cosine at `step=end_step` equals `min_lr` within float tolerance.
- `--lr-schedule cosine --min-lr 0` decays to zero at the end.
- Resuming at `start_step > warmup_steps` does not restart warmup.
- `--lr-schedule cosine --lr-decay 1` exits before training.
- `--lr-schedule step` reproduces the old behavior bit-for-bit.

### Tests and docs

- Add `tests/unit/tools/unit_lr_schedule.c` with boundary tests and register it in
  `tests/test_main.c`.
- Compare a few schedule points against a short Python/numpy reference and record the
  expected values in the test.
- Document the schedule in `docs/CLI_REFERENCE.md` with a small table of step -> LR.
- Update the production config example to use warmup + cosine once validated.

### Risks

- Changing the default schedule would invalidate existing recipes. Keep the default
  `step` behavior and make cosine opt-in.
- `end_step` is `start_step + steps`; on resume, the cosine span is the remaining run, not
  the original run. This is deliberate but should be documented. If exact resume is needed,
  store the original `end_step` in the optimizer metadata (item 10).
- Float rounding at the cosine endpoint may produce a tiny nonzero LR. Clamp `progress` to
  `[0, 1]`.

---

## 4. [DONE] Training telemetry (params, throughput, MFU)

**Status:** not started  
**Effort:** 0.5-1 day  
**Depends on:** item 3 for the LR value in the report

### Why

nanoGPT prints parameters at model construction (`get_num_params()`), tokens per iteration,
wall time per iteration, and a smoothed MFU estimate (`estimate_mfu`). fBLT prints the
config at 574 but not the parameter count; `[TIMING]` is hidden behind the
`BLT_TRAIN_TIMING` environment variable; and `RESULT` has no throughput or model-size
summary. These are basic run-comparison numbers.

### Current state

- `blt_count_params()` already exists at `csrc/train_blt_d.c:274-285` and is used for the
  AdamW arena size at 385.
- `[TIMING]` at 836-842 prints fwd/bwd/opt/step milliseconds every 50 steps when
  `BLT_TRAIN_TIMING` is set.
- `core/flops.h` provides `blt_flops_per_byte()` and the per-module primitives.
- `train_sweep.c` uses `core/flops.h`; `cuda_bench` reports TFLOP/s and MFU.
- `RESULT` at 1031 only reports the config, eval windows, BPB, and masked accuracy.

### Implementation

1. Print the parameter count in the startup line at 574:

   ```c
   const size_t n_params = blt_count_params(model);
   printf("train_blt_d: ... params=%zu (%.3fM) ...",
          n_params, (double)n_params / 1e6);
   ```

   `blt_count_params` must be called after `blt_model_create` and after any BF16 copies are
   allocated; it counts logical parameters, not buffer bytes.

2. Track report-window throughput. Keep separate variables so the ETA logic at 889-899 is
   not disturbed:

   ```c
   double report_t0 = blt_time_sec();
   size_t report_step0 = a.start_step;
   size_t patch_sum = 0, patch_n = 0;
   ```

   At each report:

   ```c
   const double now = blt_time_sec();
   const size_t steps_done = (step + 1) - report_step0;
   const double dt = now - report_t0;
   const double ms_per_step = 1000.0 * dt / (double)steps_done;
   const double bytes_per_s = (double)steps_done * (double)a.window / dt;
   const double avg_patch = patch_n ? (double)patch_sum / (double)patch_n : 4.0;
   ```

   Reset `report_t0`, `report_step0`, `patch_sum`, `patch_n` after printing.

3. Add `patch_sum += M; patch_n++;` once per training step (inside the micro-batch loop
   once item 7 lands; use the per-optimizer-step average).

4. Build a `blt_flops_config` for the forward pass:

   ```c
   blt_flops_config fc;
   memset(&fc, 0, sizeof(fc));
   fc.h_G = model->config.global_config.embed_dim;
   fc.l_G = model->config.global_config.num_layers;
   fc.n_ctx = a.window;
   fc.global_heads = model->config.global_config.num_heads;
   fc.global_head_dim = fc.h_G / fc.global_heads;
   fc.d_ff_G = model->config.global_config.hidden_dim / fc.h_G;
   fc.h_E = model->config.encoder_config.embed_dim;
   fc.l_E = model->config.encoder_config.num_layers;
   fc.w_E = a.window;
   fc.enc_heads = model->config.encoder_config.num_heads;
   fc.enc_head_dim = fc.h_E / fc.enc_heads;
   fc.d_ff_E = model->config.encoder_config.hidden_dim / fc.h_E;
   fc.h_D = model->config.decoder_config.embed_dim;
   fc.l_D = model->config.decoder_config.num_layers;
   fc.w_D = a.window;
   fc.dec_heads = model->config.decoder_config.num_heads;
   fc.dec_head_dim = fc.h_D / fc.dec_heads;
   fc.d_ff_D = model->config.decoder_config.hidden_dim / fc.h_D;
   fc.n_p = (size_t)llround(avg_patch);
   fc.k = 1;                 // no head-split factor is exposed by the trainer
   fc.vocab_size = 256;
   ```

   `blt_flops_feedforward` expects `d_ff` as a multiplier, while the model config stores an
   absolute FFN width. The divisions above are intentional. Check the integer
   divisibility; the model builder already requires `embed_dim % num_heads == 0`.

5. Estimate training FLOPs. `blt_flops_per_byte` is a forward-pass figure. A common rough
   factor for fwd+bwd is 3x, so:

   ```c
   const double flops_per_byte = blt_flops_per_byte(&fc);
   const double flops_per_step = 3.0 * flops_per_byte * (double)a.window;
   const double tflops = flops_per_step / (dt / (double)steps_done) / 1e12;
   ```

   Label the printed value as approximate (`~`). If `a.peak_tflops > 0`, print
   `mfu = 100.0 * tflops / a.peak_tflops`; otherwise omit MFU. Do not hardcode an A100 peak
   like nanoGPT does; the user may be on any GPU.

6. Add `--peak-tflops F` to `args_t` (default 0, disabled) for users who want an MFU
   number. This is optional; throughput in bytes/s and ms/step is useful without it.

7. Extend the periodic line:

   ```text
   step 1234/40000  epoch 1/10  loss 4.1234  lr 8.1e-05  ms/step 312.4  B/s 153.8  ~1.23 TFLOP/s
   ```

8. Extend `RESULT`:

   ```text
   ... params=2970000 (2.970M) best_causal_bpb=2.1012 best_step=36000
   ```

   Only include the best fields when `--save-best` is active.

9. Add the `params`/throughput fields to the JSONL `result` event (item 2).

### Acceptance

- The startup line includes the parameter count in raw and millions.
- The periodic line includes LR, ms/step, and bytes/s; the numbers are stable and sum
  correctly over a report interval.
- `--peak-tflops` produces a plausible MFU for a known GPU; without it, no MFU is printed.
- `RESULT` includes params and, when best-checkpointing is enabled, best metric/step.
- `BLT_TRAIN_TIMING` behavior is unchanged.

### Tests and docs

- No unit test is needed for the formatting, but a CI smoke run can grep the startup and
  `RESULT` lines for `params=`.
- Add a short "Reading the training log" section to the README or CLI reference.
- Verify the FLOPs estimate against `cuda_bench` on a known config; document the 3x
  fwd+bwd assumption and that cross-attention `k=1` is an approximation.

### Risks

- `blt_flops_per_byte` divides by `n_p` and `k`; a zero or fractional `n_p` can produce
  nonsense. Clamp `avg_patch` to at least 1 and skip the TFLOP line when the result is not
  finite.
- On CPU, the TFLOP/s number is not meaningful for comparing hardware but is still useful
  for relative run comparisons. Label it as an estimate.
- Counting parameters after BF16 copies exist must not double-count. Use the logical model
  tensors, as `blt_count_params` already does.

---

## 5. [DONE] Inference: `--num-samples` and `--top-k`

**Status:** not started  
**Effort:** 0.5-1 day  
**Depends on:** nothing

### Why

nanoGPT's `sample.py` exposes `num_samples` and `top_k` alongside `temperature`
(`sample.py:14-19`). fBLT has temperature, top-p, repetition penalty, and no-repeat-ngram,
which is a richer sampler overall, but it lacks top-k and can only draw one sample per
invocation. Both are common enough that their absence is noticeable.

### Current state

- `csrc/core/decode_select.h` defines `blt_decode_options` with `temperature`, `top_p`,
  `repeat_penalty`, `no_repeat_ngram`, `seed`.
- `blt_decode_is_sampling()` at `csrc/core/decode_select.c:18` checks those four knobs.
- `blt_decode_select()` at `csrc/core/decode_select.c:49` applies bans, penalty,
  temperature, softmax, then top-p.
- `csrc/infer.c`:
  - `infer_args` has `temperature`, `top_p`, `repeat_penalty`, `no_repeat_ngram`,
    `sampling_given`.
  - `dopts` is populated at 390-396.
  - The greedy dispatch is at 533-542.
  - The output buffer is allocated at 522-523 and written once at 602-605.
- `fblt/_config.py` `InferConfig` exposes the same fields.

### Implementation

#### 5a. Top-k

1. Add `size_t top_k;` to `blt_decode_options` and default it to 0 in
   `blt_decode_options_defaults`.

2. Update `blt_decode_is_sampling`:

   ```c
   return o->temperature > 0.0f || o->top_p > 0.0f || o->top_k > 0 ||
          o->repeat_penalty != 1.0f || o->no_repeat_ngram >= 2;
   ```

3. Update the order-of-operations comment in `decode_select.h`: bans, penalty,
   temperature, softmax, top-k, top-p, draw.

4. Restructure the tail of `blt_decode_select`. The current top-p branch sorts `order`
   descending and finds the nucleus. Reuse that sort for top-k:

   ```c
   size_t keep = vocab_size;
   if (o->top_k > 0 && o->top_k < keep) keep = o->top_k;
   if (o->top_p > 0.0f && o->top_p < 1.0f) {
       // existing insertion sort into order[]
       double cum = 0.0;
       size_t nucleus = vocab_size - 1;
       for (size_t i = 0; i < vocab_size; i++) {
           cum += work[order[i]];
           if (cum >= (double)o->top_p) { nucleus = i; break; }
       }
       if (nucleus + 1 < keep) keep = nucleus + 1;
   }
   ```

   If `keep < vocab_size`, zero every probability outside the first `keep` entries in
   `order`, renormalize, and draw from the prefix. If `keep == 1`, return `order[0]`
   directly (this must equal argmax for `top_k=1`).

   Avoid sorting when neither `top_k` nor `top_p` is active, to keep the fast path.

5. Add `--top-k N` to `csrc/infer.c`'s `usage()` and parser. Set `dopts.top_k = a.top_k`.
   Add `top_k: int = 0` to `InferConfig`.

6. Semantics: `top_k=0` disables. `top_k=1` is greedy. If both `top_k` and `top_p` are
   set, apply top-k first, then nucleus on the remaining distribution. Document this.

#### 5b. `--num-samples`

1. Add `size_t num_samples;` to `infer_args`, default 1, and `--num-samples N` to the
   parser/usage.

2. Restrict it to the greedy sampling path. `selfspec`, `blockdiff`, and `blockdv` are
   deterministic (or use their own selection logic), so `--num-samples > 1` there would
   produce duplicates. Fail early:

   ```c
   BLT_REQUIRE(a.num_samples == 1 || a.method == M_GREEDY,
               "--num-samples > 1 is only supported with --method greedy");
   ```

3. Move the output-file open before the generation loop. Reuse the `output` buffer:
   `total_out = prompt_len + a.new_bytes` is already the right size.

4. Loop around the method dispatch:

   ```c
   for (size_t s = 0; s < a.num_samples; s++) {
       if (a.num_samples > 1) {
           dopts.seed = a.seed + s;   // reproducible per-sample stream
           fprintf(stderr, "[infer] sample %zu/%zu seed=%llu\n",
                   s + 1, a.num_samples, (unsigned long long)dopts.seed);
       }
       memset(&st, 0, sizeof(st));
       switch (a.method) { ... }
       // write prompt + generated bytes
       fwrite(output, 1, total_out, outfp);
       if (a.num_samples > 1 && !a.output) {
           fprintf(outfp, "\n----- sample %zu -----\n", s + 1);
       }
       // aggregate stats
   }
   ```

   For `num_samples == 1`, the byte output must be identical to the current path. Keep the
   existing no-separator behavior for one sample.

5. Aggregate `blt_infer_stats`: sum `nfe_encoder_global`, `nfe_decoder`, `bytes_drafted`,
   `bytes_ar`, `bytes_accepted` across samples, then print one aggregate line. If the
   stats struct has an accumulate helper, use it; otherwise add a small local accumulator.
   Print per-sample lines only in verbose/debug mode.

6. Add `--num-samples` and `--top-k` to `fblt/_config.py` and `docs/CLI_REFERENCE.md`.

### Acceptance

- `--top-k 1` with `--temperature 0` produces byte-identical output to plain greedy.
- `--top-k 1` with `--temperature 0.8` also produces the argmax byte (top-k of one).
- `--top-k 40 --temperature 0.8` changes the distribution and remains stable across seeds.
- `--num-samples 3` prints three separated samples to stdout, or writes three concatenated
  samples to `--output`, with reproducible seeds `seed`, `seed+1`, `seed+2`.
- `--num-samples 3 --method blockdv` fails with a clear message.
- `--top-k` is ignored by non-greedy methods exactly as `--temperature` is today (they
  already print/validate sampling options; keep the existing behavior consistent).

### Tests and docs

- Extend `tests/unit/tools/unit_topk.c` and `run_topk_from_logits_tests` with decode-select
  cases: top-k=1 equals argmax, top-k filters the tail, top-k+top-p composes, all-banned
  fallback still works.
- Add an integration test that runs `infer --top-k 1` and compares the output bytes to
  `infer` greedy with the same prompt/seed.
- Add a Python test that `InferConfig.to_argv()` emits `--top-k` and `--num-samples`.
- Document the sampling order in `docs/CLI_REFERENCE.md` and add a `num_samples` example to
  the README.

### Risks

- The top-p branch currently allocates `order` only when top-p is active. If you unify it
  with top-k, ensure the free paths are correct on every early return.
- `blt_decode_select` uses a 256-entry vocab; an O(n^2) insertion sort is fine. Do not
  replace it with a general-purpose sorter that changes tie behavior without a test.
- Writing multiple samples to `--output` changes the file format for that flag. Document
  it; users who want separate files can loop the CLI or use the wrapper.

---

## 6. [DONE] Profiling target + docs

**Status:** not started  
**Effort:** 0.5-1 day  
**Depends on:** nothing

### Why

nanoGPT's `bench.py` has a `--profile` path using `torch.profiler` with a TensorBoard
trace handler (`bench.py:66-95`). fBLT has `[TIMING]` and the CUDA benchmark harness, but
the profiling recipe is not documented, and `AGENTS.md` points at a `docs/Plan.md` that does
not exist. A one-command nsys wrapper makes performance work repeatable.

### Current state

- `csrc/train_blt_d.c:836-842` prints `[TIMING] step ... fwd ... bwd ... opt ... step ...`
  every 50 steps when `BLT_TRAIN_TIMING` is set.
- `bench/cuda_bench.c`, `bench/infer_bench.c`, and `bench/harness.c` exist.
- `scripts/` contains only `cuda_env.sh`.
- `docs/` has no profiling guide. `AGENTS.md` references `docs/Plan.md`, which is absent.
- `CMakeLists.txt` has benchmark targets (`bench-infer`, `bench-cuda`, `bench-harness`) but
  no profiling target.

### Implementation

1. Add `scripts/profile_train.sh`:

   ```sh
   #!/usr/bin/env bash
   set -euo pipefail
   build_dir="${1:-build-cuda}"
   shift || true
   out_dir="${OUT_DIR:-runs/profiles}"
   mkdir -p "$out_dir"
   stamp="$(date +%Y%m%d-%H%M%S)"
   nsys profile \
     --trace=cuda,nvtx,osrt \
     --sample=none \
     --cpuctxsw=none \
     -o "$out_dir/train_$stamp" \
     "$build_dir/train_blt_d" "$@"
   echo "report: $out_dir/train_$stamp.nsys-rep"
   echo "stats:  nsys stats $out_dir/train_$stamp.nsys-rep"
   ```

   Make it executable. Reject a missing `nsys` with an install hint. Pass all remaining
   arguments to `train_blt_d`, so the caller supplies `--corpus`, `--steps`, etc.

2. Add `docs/PROFILING.md` covering:

   - CPU: `BLT_TRAIN_TIMING=1 build/train_blt_d ...`, `perf record`/`perf report`.
   - CUDA: the `nsys profile` command above, `nsys stats --report cuda_gpu_kern_sum`,
     `ncu --set full` for a single kernel, and `compute-sanitizer` for correctness.
   - How to read `[TIMING]` (fwd/bwd/opt/step).
   - A note that `[TIMING]` is every 50 steps and averages the window, not per-step
     percentiles.
   - The benchmark targets: `bench-infer`, `bench-cuda`, `bench-harness`.

3. Add a wrapper flag to `fblt/train.py`:

   ```text
   --profile    run under nsys profile (CUDA only), write runs/<name>/profile.nsys-rep
   ```

   Implementation:

   ```python
   if args.profile:
       if cfg.backend != "cuda":
           raise SystemExit("--profile requires --backend cuda")
       nsys = shutil.which("nsys")
       if nsys is None:
           raise SystemExit("nsys not found; install Nsight Systems or run the binary directly")
       cmd = [
           nsys,
           "profile",
           "--trace=cuda,nvtx,osrt",
           "--sample=none",
           "--cpuctxsw=none",
           "-o",
           os.path.join(run_dir, "profile"),
           binary,
       ] + argv
       return subprocess.call(cmd)
   ```

   Keep `--profile` out of `TrainConfig`; it is a wrapper concern.

4. Optionally add a CMake target that prints usage rather than hardcoding a corpus:

   ```cmake
   add_custom_target(profile-train
     COMMAND ${CMAKE_COMMAND} -E echo "usage: scripts/profile_train.sh build-cuda --corpus ..."
     COMMENT "Profiling wrapper")
   ```

   A target that requires arguments is awkward; the script is the primary interface. If
   you add the target, make it a help target, not an auto-run.

5. Fix the stale reference in `AGENTS.md`: `docs/Plan.md` -> `docs/PROFILING.md`.

### Acceptance

- `scripts/profile_train.sh build-cuda --corpus data/train.bin --steps 50 ...` produces a
  `.nsys-rep` and prints the `nsys stats` command.
- `fblt-train --profile --backend cuda ...` writes the report under the run directory and
  still streams training output.
- `docs/PROFILING.md` commands work as written on Linux with CUDA; the CPU section works
  without CUDA.
- `AGENTS.md` no longer references a missing file.

### Tests and docs

- CI cannot run nsys without a GPU; add a shell syntax check (`bash -n
  scripts/profile_train.sh`) and a Python test for the wrapper's backend/nsys validation.
- Link the profiling doc from the README and from `docs/CLI_REFERENCE.md`.
- Add a short note to `CONTRIBUTING.md`: performance PRs should include the nsys command
  and the `[TIMING]` summary.

### Risks

- `nsys` output files are large and are already gitignored (`*.nsys-rep`, `*.sqlite`).
- NVTX ranges are only useful if the code emits them. The current trainer does not; the
  `osrt`/CUDA traces still show kernel and API time. If you want labeled stages, add
  `nvtxRangePushA`/`nvtxRangePop` around forward/backward/optimizer under a compile guard.
  That is optional and should not be required for the first version.
- Windows profiling is out of scope; document Linux-only.

---

# 1.x

## 7. Gradient accumulation / batch > 1

**Status:** not started  
**Effort:** 3-7 days  
**Depends on:** items 1, 2, and 4 for useful logging; item 9 recommended first so the
optimizer recipe is stable

### Why

fBLT trains one window per optimizer step. nanoGPT supports `batch_size` and
`gradient_accumulation_steps`, which is how it reaches a ~0.5M-token effective batch on
8 GPUs (`config/train_gpt2.py`). Without accumulation, fBLT cannot reproduce large-batch
recipes, and gradient noise per step is high. This is the biggest training-capability gap.

### Current state

- `csrc/train_blt_d.c:642` picks one window:
  `const size_t w = train_window_index(step, num_windows, a.seed);`.
- The loop body from 643 to 963 does one forward, one backward, clip, and one optimizer
  step.
- Gradients live in `grad`, created from `model_arena` at 418:
  `blt_model_grad *grad = blt_model_grad_create(model_arena, model);`.
- Grad zeroing at 686-691 explicitly zeros the encoder embedding, ngram tables, decoder
  d0 embedding, and all norm gradients via `blt_model_visit_params(... zero_norm_fn ...)`.
- Backward functions are called once per step:
  `blt_local_decoder_backward_diffusion`, `blt_global_transformer_backward`,
  `blt_local_encoder_backward`, or `blt_model_backward` for the plain path.
- `clip_all` and `adamw_all`/`sgd_all` take the single `grad`.
- `args_t` has no batch/accumulation field.

### Implementation

Use `--grad-accum N` as the primary flag. For this model, one window is the micro-batch, so
N accumulated windows are equivalent to a batch of N. Do not add a separate `--batch-size`
until there is a real multi-window batch representation; two flags with the same meaning
will confuse users.

1. Add `size_t grad_accum;` to `args_t`, default 1. Parse `--grad-accum N` and validate
   `N >= 1`. Add it to `TrainConfig` and the CLI docs.

2. Add a second gradient buffer, allocated from `model_arena`:

   ```c
   blt_model_grad *grad_accum = blt_model_grad_create(model_arena, model);
   ```

   This doubles gradient memory. For the 7.8M-parameter model that is roughly 31 MB in
   FP32, fine for the current 64 MB CPU arena and 1 GB CUDA arena. For larger models,
   increase `--model-mb` or allocate the accumulator from a dedicated arena. Document the
   memory cost.

3. Add reusable gradient helpers. The current explicit zero list plus `zero_norm_fn` is
   fragile; centralize it:

   ```c
   void blt_model_grad_zero(const blt_model *model, blt_model_grad *g);
   void blt_model_grad_add(const blt_model *model, blt_model_grad *dst,
                           const blt_model_grad *src);
   void blt_model_grad_scale(const blt_model *model, blt_model_grad *g, float scale);
   ```

   Implement them in `csrc/models/model.c` or a new `csrc/models/grad_ops.c`. Use
   `blt_model_visit_params(model, g, fn, ctx, 1)` for zero/scale. For add, collect source
   pointers by `info->idx` in a first pass, then add into the destination in a second pass
   over `blt_model_visit_param_pairs(model, dst, ...)`. The visitor guarantees a stable
   `idx` across calls (`csrc/models/param_visitor.h:20-40`).

   Before replacing the explicit zeros, verify that the visitor covers the encoder
   embedding, ngram tables, decoder d0 embedding, and all norm gradients. If it does, the
   explicit calls can be removed; if not, keep them and add the missing tensors to the
   helper.

4. Restructure the training loop:

   ```c
   blt_model_grad_zero(model, grad_accum);
   double step_loss = 0.0;
   size_t step_patches = 0;

   for (size_t j = 0; j < a.grad_accum; j++) {
       const size_t data_step = step * a.grad_accum + j;
       const size_t w = train_window_index(data_step, num_windows, a.seed);

       // reset scratch and all per-micro-batch state
       blt_arena_reset(scratch);
       // build patches, batch, bytes_in, targets
       // zero `grad` (not grad_accum) before backward
       blt_model_grad_zero(model, grad);

       // forward + backward into `grad`
       // ...
       step_loss += lv;
       step_patches += M;

       blt_model_grad_add(model, grad_accum, grad);
   }

   blt_model_grad_scale(model, grad_accum, 1.0f / (float)a.grad_accum);

   const float lr = train_lr_at_step(&a, step, end_step);
   const float pre_clip_norm = clip_all(model, grad_accum, a.max_norm);
   // diagnostics and optimizer use grad_accum
   adamw_all(model, grad_accum, aw, &cfg);   // or sgd_all
   ```

   The order matters: scale after accumulation, before clipping, so the clip threshold
   applies to the averaged gradient, matching a true batch of N.

5. Keep `blt_model_grad_zero(model, grad)` inside each micro-batch. Backward functions may
   accumulate some gradients (norm weights are zeroed separately today), so a clean source
   buffer is required. Do not zero `grad_accum` until the start of the next optimizer step.

6. Data order and schedules:

   - Window index: `train_window_index(step * grad_accum + j, num_windows, a.seed)`.
     For `grad_accum=1` this is exactly the old `train_window_index(step, ...)`.
   - Block-diffusion timestep RNG: replace `a.seed * 0x9E3779B97F4A7C15ULL + step + 1` with
     `... + data_step + 1`.
   - `blt_block_batch_build_t(..., a.seed + step, ...)`: use `a.seed + data_step`.
   - Mask warmup, `--t-warmup-hi`, and LR schedules stay on the optimizer `step`, not
     `data_step`, because they are defined in optimizer steps. Document this distinction.
   - Epoch count: `total_epochs = (start_step + steps) * grad_accum / num_windows`
     (round up). The periodic report's `epoch = (step * grad_accum) / num_windows`.

7. Resume:

   - `--start-step` is an optimizer step. The first micro-batch is
     `start_step * grad_accum`.
   - Verify that the window order and RNG streams after resume match an uninterrupted run.
     Add a regression test (below).
   - `--save-every` and best-checkpoint steps remain optimizer steps.

8. Diagnostics:

   - `--loss-log`: log the averaged `step_loss / grad_accum` once per optimizer step, not
     once per micro-batch.
   - `--grad-norm-log`, `--update-norm-log`, `--component-norm-log`: use `grad_accum`, after
     scaling and clipping.
   - `--activation-dump-log`: dumping every micro-batch multiplies the log size. Dump only
     the last micro-batch, or add a separate interval.
   - `[MONITOR]` and `[EVAL]`: run once per optimizer step, after the optimizer update.
   - `[TIMING]`: divide the step time by the number of optimizer steps, not micro-batches.

9. `--report-every`, ETA, and `running/running_n` should count optimizer steps, not
   micro-batches. `running` should add the averaged step loss.

### Acceptance

- `--grad-accum 1` is bit-identical to the current trainer for the same seed, config, and
  step count. This is the non-negotiable regression check.
- `--grad-accum 2` produces gradients equal to the average of two single-window gradients
  from the same windows before any optimizer update. Test this with SGD and a fixed tiny
  model.
- A 10-step run with `--grad-accum 2`, saved and resumed at step 5, matches an uninterrupted
  10-step run (loss and weights bit-identical, or within documented float tolerance).
- Epoch/ETA/report counts are correct: after `steps` optimizer steps with accumulation N,
  the trainer has consumed `steps * N` windows.
- Memory use grows by roughly one gradient buffer; `--model-mb` guidance is documented.

### Tests and docs

- Add `tests/unit/models/unit_grad_accum.c`:
  - zero/add/scale helpers cover every tensor in `blt_model_grad`;
  - add then scale equals the mean of two gradients;
  - zero restores all gradients to zero.
- Add a CLI integration test: train 2 steps with `--grad-accum 2`, compare the resulting
  checkpoint to a manually averaged reference (or to a 2-step run with a fixed data order).
- Add a resume test: save at optimizer step 3, resume, compare to uninterrupted.
- Document the flag, the memory cost, the data-order semantics, and the schedule semantics
  in `docs/CLI_REFERENCE.md`.
- Add a `configs/train/production_accum.yaml` example once validated.

### Risks

- **Backward semantics:** if any backward function assigns rather than accumulates, using a
  clean `grad` per micro-batch is correct. If it accumulates internally, zeroing before
  each micro-batch is correct. Audit each of the four backward entry points before relying
  on the helper; add a test that a second backward without zeroing changes the gradient as
  expected.
- **Norm gradient zeroing:** the current code zeroes norm gradients separately because the
  backward path accumulates them. The new `blt_model_grad_zero` must preserve that. A
  missed tensor produces silently wrong gradients that still look plausible.
- **Arena growth:** `model_arena` is sized 64 MB on CPU. Doubling gradients plus AdamW
  moments can overflow for larger configs. Add a clear allocation error or derive the arena
  size from the parameter count, as the AdamW arena already does.
- **Dropout/RNG:** if dropout is added later, the RNG stream must be per-micro-batch, not
  per-optimizer-step. Document the RNG ownership.
- **Performance:** accumulating in FP32 on device costs one extra read/write per parameter
  per micro-batch. Measure before assuming it is free; the add can be fused into the
  backward epilogue later.

---

## 8. BF16 mixed-precision training

**Status:** partially wired, not usable end-to-end  
**Effort:** 1-3 weeks  
**Depends on:** item 9 (stable optimizer recipe) and item 10 (checkpoint metadata)

### Why

The README reports BF16 mixed-precision matmuls at roughly 20 TFLOP/s versus 6.4 TFLOP/s
FP32 on an RTX 4060. The kernels and BF16 weight-copy plumbing exist, but the trainer does
not expose them and the local encoder/decoder do not use them. Wiring this is the largest
single-device speed win available.

### Current state

- `BLT_DTYPE_BF16` exists in `csrc/core/dtype.h`.
- `blt_cast` supports FP32 <-> BF16 on CPU (`csrc/ops/cast_cpu.c`) and CUDA
  (`csrc/ops/cast_cuda.cu`).
- `blt_matmul` supports BF16 inputs and uses `cublasGemmEx` with FP32 accumulation on CUDA
  (`csrc/ops/linalg.cu:46-51`). The CPU path converts BF16 to FP32 temporaries
  (`csrc/ops/linalg_cpu.c:23-42`), so CPU BF16 is not a speed path.
- `blt_transformer_stack_config` has `use_bf16` (`csrc/models/transformer_stack.h:49`).
- `blt_transformer_stack_init` allocates BF16 copies of the five matmul weights
  (`transformer_stack.c:87-132`) and uses them in forward (`transformer_stack.c:214-217`).
- `blt_mixed_weight_refresh(bf16_copy, fp32_master)` exists
  (`transformer_stack.c:23-28`) but is **never called** anywhere.
- Only `blt_global_transformer_config` exposes `use_bf16`
  (`global_transformer.h:26`); the local encoder and decoder configs do not.
- `blt_model_config_defaults` memsets the config to zero, so `use_bf16` is always 0.
- The trainer has no `--bf16`/`--dtype` flag.
- The optimizer and checkpoint code operate on FP32 master weights; BF16 copies are not
  serialized.

### Implementation

Do this in phases. Do not enable BF16 by default.

#### Phase 1: global transformer only

1. Add `--bf16` to `args_t` (or `--dtype bf16|fp32`; a boolean is enough for now).

2. Before `blt_model_create`, set:

   ```c
   cfg.global_config.use_bf16 = a.use_bf16;
   ```

3. After `blt_model_create` and after loading weights, call a new
   `blt_model_refresh_bf16(model)` that walks the global stack and calls
   `blt_mixed_weight_refresh` on each FP32/BF16 pair. The current code only allocates the
   copies; without a refresh they contain whatever the arena initialized, so global BF16
   would produce garbage.

4. After every optimizer step, call `blt_model_refresh_bf16(model)`. This copies the updated
   FP32 master weights into the BF16 views. If this is too slow, refresh every N steps and
   document the approximation; correctness first.

5. Gate on CUDA:

   ```c
   BLT_REQUIRE(!a.use_bf16 || a.use_cuda, "--bf16 requires --backend cuda");
   ```

   The CPU BF16 matmul path allocates temporaries and is not faster.

6. Add a numerical parity test: 10 steps FP32 vs 10 steps BF16 on the same seed/data with a
   small model. Loss should track within a documented tolerance; no NaN/Inf. Compare the
   final checkpoint tensors with a relative tolerance, not bit equality.

#### Phase 2: local encoder and decoder

7. Add `use_bf16` to `blt_local_encoder_config` and `blt_local_decoder_config`.
   Propagate it to their internal transformer stacks. Inspect
   `csrc/models/local_encoder.c` and `csrc/models/local_decoder.c` to see whether they use
   `blt_transformer_stack_init` or their own layer storage. If they use their own storage,
   add BF16 copy tensors and the same view/refresh pattern as `transformer_stack.c`.

8. Extend `blt_model_refresh_bf16` to cover encoder and decoder layers.

9. Re-run the parity and loss-curve tests. The local layers dominate the parameter count
   and compute, so this is where the speedup should appear.

#### Phase 3: training-loop integration and packaging

10. Add `--bf16` to `TrainConfig` and the CLI docs. Print the active dtype in the startup
    line and `RESULT`.

11. Ensure checkpoint save/load uses the FP32 masters only. BF16 copies are reconstructed
    after load; they are not part of the file. Add a test that a BF16-trained checkpoint
    loads into an FP32 model and produces the same outputs as the BF16 run within
    tolerance.

12. Account for memory: BF16 copies add ~2 bytes per matmul parameter, in addition to the
    FP32 master and AdamW moments. Document the `--model-mb` impact.

13. Consider a `--bf16-weights-only` vs `--bf16-full` distinction. The minimal safe version
    keeps activations, norms, softmax, CE, and optimizer state in FP32 and uses BF16 only
    for the matmul inputs. Do not cast the loss or AdamW state to BF16.

### Acceptance

- `--bf16` on CUDA reduces step time on a matmul-bound config without changing the loss
  curve beyond the documented tolerance.
- `--bf16` on CPU exits with a clear error (or is silently ignored with a warning; pick one
  and document it).
- A BF16 run produces no NaN/Inf over at least a few hundred steps.
- The BF16 weight copies are refreshed after optimizer steps; a unit test mutates a master
  weight and checks the BF16 copy changes after refresh.
- Checkpoints remain FP32 and load into both FP32 and BF16 models.

### Tests and docs

- Add a unit test for `blt_mixed_weight_refresh` and for `blt_model_refresh_bf16` covering
  every layer type.
- Add an integration test comparing 5-10 steps of FP32 vs BF16 loss.
- Add a CUDA benchmark entry to `bench/results.jsonl` with the config, GPU, and dtype.
- Document the flag, the numerical expectations, and the CPU limitation in
  `docs/CLI_REFERENCE.md` and `README.md`.
- Add a `--peak-tflops` example for MFU measurement (item 4).

### Risks

- **Stale BF16 copies:** the existing code allocates them but never refreshes. If you enable
  `use_bf16` without a refresh, training silently uses uninitialized or stale weights. This
  is the single most likely failure mode.
- **CPU path is slower:** `linalg_cpu.c` converts BF16 to FP32 per matmul. Do not advertise
  CPU BF16 speed.
- **Numerical instability:** BF16 has 8 exponent bits but 7 mantissa bits. LayerNorm/RMSNorm,
  softmax, and cross-entropy should stay FP32. If loss diverges, first check whether the
  optimizer state or norm weights were accidentally cast.
- **Checkpoint compatibility:** BF16 copies must not be serialized. If you add them to the
  checkpoint, loading into FP32 will fail or silently waste memory.
- **Windows CUDA:** `__nv_bfloat16` and `cublasGemmEx` are supported, but the CI job is
  compile-only. Add a runtime check on a real GPU before claiming support.

---

## 9. [DONE] Selective weight decay

**Status:** not started  
**Effort:** 0.5 day  
**Depends on:** nothing; do this before item 7/8 so accumulated/BF16 runs use the final
optimizer semantics

### Why

nanoGPT builds two optimizer groups: matrices/embeddings get weight decay, biases and
LayerNorm weights do not (`model.py:263-287`). fBLT's `adamw_all` applies `weight_decay` to
every parameter pair, including RMSNorm weights. Decaying norm scales is usually harmful or
at least non-standard, and it makes fBLT's AdamW recipe differ from the reference without a
documented reason.

### Current state

- `csrc/train_optim.c:99-125` `adamw_pair_fn` does:

  ```c
  const float wd = c->cfg.weight_decay;
  ...
  w[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * w[i]);
  ```

  for every tensor.
- `csrc/train_optim.c:141-170` `adamw_unorm_fn` repeats the same update for the norm
  estimate used by `--update-norm-log`.
- `blt_param_info` already has `is_norm`, `is_enc`, `is_glob`, `is_dec`, `layer_idx`,
  `param_in_layer`, `component_group`, `numel`, and `backend`
  (`csrc/models/param_visitor.h:20-35`).
- There are no bias tensors in the current checkpoint tensor list, so `is_norm` is the only
  no-decay class today. If biases are added later, extend the predicate.

### Implementation

1. Add `int decay;` to `blt_param_info` for clarity, or use `!info->is_norm` directly. A
   field is better because future no-decay classes (biases, scalar gains) can set it in one
   place.

2. In `csrc/models/param_visitor.c`, `fill_info` sets `info->decay = !info->is_norm;`
   (and `&& !is_bias` if a bias class is added).

3. In `adamw_pair_fn`:

   ```c
   const float wd = info->decay ? c->cfg.weight_decay : 0.0f;
   ```

4. In `adamw_unorm_fn`, use the same predicate so the update-norm log matches the actual
   update.

5. Add a `--weight-decay-norms` compatibility flag if you need to reproduce old runs
   exactly. Default it off. If you do not add the flag, document the change in
   `CHANGELOG.md` and re-run any reference recipe.

6. Do not change SGD; it has no weight decay.

### Acceptance

- A zero-gradient AdamW step with `weight_decay > 0` leaves all `is_norm` tensors
  bit-identical and shrinks matrix/embedding tensors by `lr * wd * w`.
- `--update-norm-log` matches the actual parameter update for both groups.
- Existing FP32 training with `--weight-decay 0` is unchanged.
- A short training run with the selective recipe produces finite losses and a sane BPB
  curve.

### Tests and docs

- Add a unit test that creates a tiny model, sets one norm weight and one matrix weight to
  a known value, runs one AdamW step with zero gradients, and asserts the norm is unchanged
  while the matrix decays.
- Add a test for `adamw_update_norm` consistency.
- Document the two param groups in `docs/CLI_REFERENCE.md` and the optimizer section of the
  README.
- Note the behavior change in `CHANGELOG.md`.

### Risks

- This changes training dynamics for every AdamW run. If existing recipes were tuned with
  norm decay, they may need a small LR adjustment. Run the reference config before and
  after and record the difference.
- The `blt_param_info` visitor is shared by diagnostics, clipping, and checkpoint code. Add
  the field without changing the iteration order or `idx` assignment; those are load-bearing
  for AdamW state pairing and gradient accumulation.

---

## 10. [DONE] Checkpoint metadata v2

**Status:** not started  
**Effort:** 1-2 days  
**Depends on:** nothing; item 1 benefits from persisted best-metric state

### Why

nanoGPT embeds `model_args`, `iter_num`, `best_val_loss`, and the full config in
`ckpt.pt` (`train.py:277-284`). fBLT's `.fblt` format stores tensor names and shapes but not
the architecture config, step, or best metric. The Python wrapper compensates by walking up
from the checkpoint to find `resolved_config.yaml`, and `infer` requires the user to pass
`--embed`, `--hidden`, and layer counts. A checkpoint moved without its sidecar is not
self-describing.

### Current state

- `csrc/models/checkpoint.h` documents format v1: magic, `u32 version`, `u32 num_tensors`,
  then per-tensor name/dims/data.
- `csrc/models/checkpoint.c:177-207` writes v1; `308-350` reads v1 and rejects any version
  other than 1 (`checkpoint.c:260`, `316`).
- `csrc/models/checkpoint.c:210-250` writes entropy-LM weights with the same v1 container.
- `fblt/infer.py:82-120` finds `resolved_config.yaml` by walking parent directories and
  fills shape/patching fields from it.
- `fblt/train.py` writes `runs/<name>/resolved_config.yaml`.
- The optimizer state file is separate and already stores a step counter
  (`csrc/train_optim.c:175-210`).

### Implementation

#### Format

Keep the magic and bump the version:

```text
offset 0:  char magic[4] = "FBLT"
offset 4:  u32 version = 2
offset 8:  u32 meta_len
offset 12: u8  meta[meta_len]       // UTF-8 JSON, not NUL-terminated
offset 12+meta_len: u32 num_tensors
then the existing per-tensor records
```

All integers and floats are little-endian, matching the current file. Use a fixed upper
bound on `meta_len` (for example 1 MiB) before allocating.

#### Metadata contents

```json
{
  "format": "fblt-checkpoint",
  "format_version": 2,
  "created_by": "fBLT 1.0.0",
  "git_commit": "9b7f5ac",
  "config": {
    "embed": 192,
    "hidden": 384,
    "enc_layers": 2,
    "glob_layers": 2,
    "dec_layers": 2,
    "cross_attn": "all",
    "d0_mode": "learned",
    "diffusion": 1,
    "block_size": 4,
    "window": 48,
    "entropy_patches": false,
    "ngram_sizes": [3, 4]
  },
  "training": {
    "step": 40000,
    "optimizer": "adamw",
    "seed": 7,
    "best_metric": "causal_bpb",
    "best_value": 2.1012,
    "best_step": 36000
  }
}
```

Omit unknown/optional fields rather than writing `null`; the reader should treat missing
fields as unknown.

#### C API

1. Add:

   ```c
   void blt_model_save_ex(const blt_model *model, const char *path,
                          const char *meta_json);
   void blt_model_load_ex(blt_model *model, const char *path,
                          char *meta_out, size_t meta_cap);
   int blt_model_read_meta(const char *path, char *meta_out, size_t meta_cap);
   ```

   `blt_model_save` becomes a wrapper that calls `_ex` with a minimal metadata blob
   (or `NULL`, in which case write an empty JSON object). `blt_model_load` becomes a
   wrapper that discards metadata.

2. `blt_model_save_ex` writes the v2 header, then the existing tensor loop.

3. `blt_model_load_ex` reads magic and version:

   - version 1: use the existing path; return 0 for "no metadata".
   - version 2: read `meta_len`, validate it, copy metadata to `meta_out` if it fits,
     then read `num_tensors` and validate as before.
   - any other version: `BLT_FATAL`.

4. `blt_model_read_meta` reads only the header and metadata; it does not build a model. Use
   it in `infer` and in a future `model_info` tool.

5. Keep tensor names and shapes unchanged. The metadata is additive; a v1 reader cannot
   read v2, but that is expected. A v2 reader must read v1.

6. Build the metadata JSON on the C side with a small `snprintf` helper. The fields are
   simple scalars and fixed strings; escape any user-provided string (paths, commit) or
   omit them. A 4 KiB stack buffer is enough; assert on truncation.

7. For entropy-LM checkpoints, either keep v1 or add the same v2 header later. Do not
   change both formats in one step unless the reader is shared and tested.

#### Consumers

8. `csrc/train_blt_d.c`: replace `blt_model_save` with `blt_model_save_ex` at the periodic,
   final, and best-checkpoint sites, passing a metadata JSON built from `a`, `model->config`,
   the current step, and the best metric. Save the same metadata in the optimizer file if
   you want resume to restore the best value; otherwise document that best tracking is
   per-process.

9. `csrc/infer.c`: make the shape flags optional when the checkpoint has metadata. Resolve
   in this order: explicit CLI flags -> metadata -> sidecar (for v1) -> error. The current
   requirement at `usage()` can become "required for v1 checkpoints".

10. `fblt/_checkpoint.py` (new): a small reader used by the Python wrapper. It parses the
    header with `struct` and `json`, supports v1 (returns `None`) and v2, and validates the
    meta length. `fblt/infer.py` uses it before the sidecar walk; `fblt/train.py` uses it to
    print the training step/architecture when loading a checkpoint.

11. Add a `model_info` target/binary that prints the metadata and tensor summary. This is
    useful for support even if no other code consumes it.

### Acceptance

- A v2 checkpoint round-trips: save, read metadata, load weights, compare outputs to the
  original model.
- A v1 checkpoint still loads unchanged. Add a checked-in v1 fixture or generate one in the
  test.
- `infer` can derive `--embed/--hidden/--enc-layers/--glob-layers/--dec-layers` from a v2
  checkpoint with no sidecar and no shape flags.
- A corrupt `meta_len` (zero, huge, truncated) fails with a clear `BLT_FATAL`, not a crash.
- The Python wrapper can read v2 metadata without invoking a binary.
- `resolved_config.yaml` remains for v1 and for human-readable run records; it is no longer
  the only source of truth.

### Tests and docs

- `tests/unit/core/unit_checkpoint.c` (or the existing checkpoint test file): v1 load, v2
  round-trip, v2 with empty metadata, corrupt length, wrong version.
- Python test for `fblt/_checkpoint.py` using a synthetic v1 and v2 header.
- Integration test: save with metadata, delete the sidecar, run `infer` with no shape flags.
- Document the format in `csrc/models/checkpoint.h`, `docs/CLI_REFERENCE.md`, and a new
  `docs/CHECKPOINTS.md`.
- State the compatibility policy: v2 readers read v1; v1 readers reject v2; the format
  version is part of the 1.0 API contract.

### Risks

- **Endianness and size assumptions:** the current format writes `size_t` for tensor dims,
  which is platform-dependent. Do not widen that problem in v2. If you want portable
  checkpoints, move dims to `u32`/`u64` in v2 and document the change. This is a good time
  to fix it, but it requires a migration path for v1.
- **Metadata size:** do not allow a 4 GB `meta_len`. Cap it and validate before `malloc`.
- **Text encoding:** JSON strings must be valid UTF-8. Keep user-provided paths out of the
  metadata or escape them.
- **Sidecar drift:** if both metadata and `resolved_config.yaml` exist and disagree, which
  wins? Explicit CLI flags first, then metadata, then sidecar. Log a warning on mismatch.

---

## 11. Multi-GPU data parallel

**Status:** design only (`MULTI_GPU.md`)  
**Effort:** 3-6 weeks  
**Depends on:** item 7 (gradient accumulation), item 9 (stable optimizer), item 10
(checkpoint metadata)

### Why

nanoGPT supports DDP on one or more nodes via `torchrun`. fBLT's `MULTI_GPU.md` describes a
data-parallel design but nothing is implemented: no NCCL, no MPI, no rank/world-size flags,
and the training loop is single-process. If 1.x is meant to scale beyond a single GPU, this
is the gate.

### Current state

- `MULTI_GPU.md` documents the current state: single GPU, single process, no NCCL/MPI,
  batch size = 1 window, 1 GB model arena + 256 MB scratch per GPU.
- `csrc/train_blt_d.c` has no rank/world-size fields.
- `csrc/core/` has no `distributed.*`.
- `CMakeLists.txt` has no NCCL option.
- Checkpoint saves and eval are unconditional (no rank guard).

### Implementation

1. Add a CMake option:

   ```cmake
   option(USE_NCCL "Enable NCCL multi-GPU support" OFF)
   if(USE_NCCL)
     find_library(NCCL_LIBRARY nccl REQUIRED)
     find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
     target_compile_definitions(blt_core PRIVATE BLT_WITH_NCCL)
     target_link_libraries(blt_core PRIVATE ${NCCL_LIBRARY})
     target_include_directories(blt_core PRIVATE ${NCCL_INCLUDE_DIR})
   endif()
   ```

   Gate to Linux + CUDA. Windows NCCL is not a supported target.

2. Add `csrc/core/distributed.h`/`.c`:

   ```c
   int  blt_dist_init(int rank, int world_size, const char *master_addr, int master_port);
   void blt_dist_finalize(void);
   int  blt_dist_rank(void);
   int  blt_dist_world_size(void);
   void blt_dist_allreduce_sum(float *buf, size_t n, int backend);
   void blt_dist_broadcast(float *buf, size_t n, int root, int backend);
   void blt_dist_barrier(void);
   ```

   Without `BLT_WITH_NCCL`, compile stubs that require `world_size == 1` and return
   identity. This keeps single-GPU builds working.

3. Add args:

   ```text
   --world-size N       number of ranks (default 1)
   --rank R             this rank, 0..N-1 (default 0)
   --master-addr IP     rendezvous address (default 127.0.0.1)
   --master-port P      rendezvous port (default 29500)
   --local-rank R       CUDA device index (default = rank)
   ```

   Also accept `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `LOCAL_RANK` from the
   environment, so a `torchrun`-style launcher works. Explicit flags win.

4. Initialize before the CUDA arena/model:

   ```c
   blt_dist_init(a.rank, a.world_size, a.master_addr, a.master_port);
   #ifdef BLT_WITH_CUDA
   if (a.use_cuda) cudaSetDevice(a.local_rank);
   #endif
   ```

   Initialize the model with the same seed on every rank (the init RNG is deterministic
   from `a.seed`), or initialize on rank 0 and broadcast the weights. The latter is safer if
   any future init path becomes rank-dependent. Broadcast every tensor once at startup.

5. Data sharding:

   ```c
   const size_t global_index = step * a.world_size + (size_t)a.rank;
   const size_t w = train_window_index(global_index, num_windows, a.seed);
   ```

   For gradient accumulation, the per-rank global index is
   `(step * a.grad_accum + j) * a.world_size + rank`. Document that ranks see different
   windows; the effective batch is `world_size * grad_accum` windows per optimizer step.
   If `num_windows` is not a multiple of `world_size`, some windows repeat across epochs;
   that is acceptable for streaming training, but note it.

6. Gradient all-reduce:

   - After the local backward and accumulation, before clipping, average gradients across
     ranks.
   - Add a helper that walks `blt_model_visit_param_pairs` and calls `ncclAllReduce` on each
     gradient pointer. Use `ncclFloat`, `ncclSum`, and the current stream.
   - After the sum, scale by `1.0 / world_size` (or use `ncclAvg` if the NCCL version
     supports it). Scaling before clip is important so the clip threshold matches the
     averaged gradient.
   - CPU tensors cannot be all-reduced with NCCL. Fatal if `info->backend !=
     BLT_BACKEND_CUDA`. If CPU multi-process is needed, add an MPI/Gloo backend separately;
     do not mix it into this item.

   ```c
   static void allreduce_pair_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
       (void)w;
       if (info->backend != BLT_BACKEND_CUDA) {
           BLT_FATAL("multi-GPU requires CUDA gradients");
       }
       ncclAllReduce(g, g, info->numel, ncclFloat, ncclSum,
                     (ncclComm_t)ctx, 0 /* current stream */);
   }
   ```

   Use the correct stream handle from the backend; do not pass 0 if the trainer uses a
   non-default stream.

7. Loss logging:

   - All-reduce the scalar step loss (sum, then divide by world size) so rank 0 prints the
     global average.
   - Or let each rank print its local loss and have rank 0 print the average. Pick one and
     document it.

8. Checkpointing:

   - Rank 0 saves weights and optimizer state. Other ranks skip and wait at a barrier.
   - All ranks load the same checkpoint from shared storage, or rank 0 loads and broadcasts.
     Shared storage is simpler; document the requirement.
   - Periodic saves and best-checkpoint saves are rank-0 only.
   - `--save-every` and `--save-best` paths must be identical across ranks.

9. Eval and monitor:

   - Rank 0 runs the held-out BPB eval and monitor; other ranks wait at a barrier.
   - If eval is expensive and you want to shard it, split windows by rank and all-reduce the
     CE sum. That is a later optimization; start with rank 0 only.

10. Determinism:

    - Same seed and same `world_size` should reproduce the same run.
    - Different `world_size` changes the data order and the effective batch; do not expect
      bit-identical loss curves.
    - Document that NCCL all-reduce is not bit-deterministic across topologies.

### Acceptance

- `--world-size 1` with `USE_NCCL=ON` is bit-identical to the current single-GPU trainer.
- A 2-GPU run with `world_size=2, grad_accum=1` uses twice the data per step and produces a
  loss curve comparable to a single-GPU run with `grad_accum=2` (within tolerance).
- Gradients after all-reduce match a CPU reference average in a unit test with a small
  buffer.
- Only rank 0 writes checkpoints/logs; no interleaved or corrupt files.
- Rank 0's eval BPB is computed from the global model, not a stale local copy.
- A rank failure terminates the job instead of hanging indefinitely (NCCL timeout/async
  error handling).

### Tests and docs

- CI without a GPU can compile-check the NCCL path with `USE_NCCL=ON` and a stub/mock, or at
  minimum run a syntax build. Runtime tests need a multi-GPU runner; document the manual
  matrix if CI cannot provide one.
- Add a single-process NCCL test (world_size=1) to the normal suite; it must be a no-op.
- Add a 2-rank integration script under `scripts/` for manual validation.
- Update `MULTI_GPU.md` from design to implementation notes, including launch commands.
- Document the effective batch formula and the shared-filesystem requirement.

### Risks

- **NCCL availability:** not present on CPU-only or Windows machines. The build must remain
  clean without it.
- **Hangs:** a rank that exits early leaves the others blocked in all-reduce. Add watchdog
  timeouts and clear error messages.
- **Memory:** each rank holds a full model, gradients, AdamW moments, and scratch. The
  current 1 GB model arena is per GPU; check it against the target model size.
- **Data collisions:** if the window index formula is wrong, ranks train on the same data
  and the effective batch does not increase. Add a test that ranks receive distinct indices
  for a small `num_windows`.
- **Checkpoint races:** two ranks writing the same path corrupts the file. Enforce rank-0
  writes and barriers.
- **Determinism claims:** do not promise bit-identical multi-GPU runs; promise reproducible
  data order and seeds.

---

## 12. [DONE] macOS support

**Status:** CI job disabled; CMake build likely close to working  
**Effort:** 2-4 days  
**Depends on:** nothing

### Why

The README lists macOS as a TODO, and `.github/workflows/ci.yml:92-100` disables the macOS
job with a note about the Makefile and OpenMP. The repository no longer has a Makefile, and
`CMakeLists.txt` does not reference OpenMP. The CMake build may already work; the disabled
job is stale. A 1.0 that claims cross-platform support should either test macOS or state
that it is unsupported.

### Current state

- No `Makefile` in the repo root.
- `CMakeLists.txt` sets `_POSIX_C_SOURCE=200809L` for non-Windows and has no OpenMP flags.
- `posix_memalign` is used in `csrc/core/allocator.c:20`; it is available on macOS.
- `clock_gettime(CLOCK_MONOTONIC)` is available on macOS 10.12+.
- CI has Linux CPU, sanitizer, CUDA syntax, clang-format, Python, Windows CPU, and Windows
  CUDA syntax jobs. The macOS job is commented out.
- CUDA is not available on modern macOS (Apple/NVIDIA support ended); CPU-only is the
  realistic target.

### Implementation

1. Re-enable a `macos-test` job in `.github/workflows/ci.yml`:

   ```yaml
   macos-test:
     runs-on: macos-latest
     timeout-minutes: 20
     steps:
       - uses: actions/checkout@...
       - name: Install dependencies
         run: brew install cmake python
       - name: Set up Python
         uses: actions/setup-python@v7
         with:
           python-version: '3.11'
       - name: Install PyTorch for parity generation
         run: pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
       - name: Generate parity data
         run: python tests/py/parity_generate.py
       - name: Configure
         run: cmake -S . -B build
       - name: Build and test
         run: cmake --build build --target test
       - name: Python tests
         run: pip install -r requirements-dev.txt && pytest tests/
   ```

   Adjust the PyTorch install to the macOS wheel; `--index-url .../cpu` works for macOS
   CPU wheels. If `torch` is too heavy for the macOS runner, cache it or limit the parity
   data generation.

2. Fix any Apple Clang issues surfaced by `-Wall -Wextra`:

   - `-Wunused-parameter` may differ from GCC. Add `(void)param;` or adjust flags.
   - `M_PI` is available with the default macOS headers, but the new LR schedule should use
     a local constant anyway.
   - `posix_memalign` is fine; `aligned_alloc` is not used.
   - `pthread` is not used by the core; the Python runner is separate.

3. Remove the stale Makefile/OpenMP comment in the CI file and replace it with a short note
   that CMake is the only build system and CUDA is unsupported on macOS.

4. Add a support-matrix section to the README:

   ```text
   Linux CPU: supported
   Linux CUDA: supported
   Windows CPU (MSYS2/MinGW): supported
   Windows CUDA: compile-checked, runtime validation required
   macOS CPU: supported/tested
   macOS CUDA: not supported
   ```

5. If the macOS job uncovers a real portability bug, fix it in the core rather than
   disabling the job. Common candidates are `ssize_t` includes, `strdup`, `PATH_MAX`, and
   `getline`.

### Acceptance

- `cmake -S . -B build && cmake --build build --target test` passes on macOS with Apple
  Clang.
- `pytest tests/` passes on macOS.
- The README support matrix matches CI.
- No CUDA/macOS claims that cannot be tested.

### Tests and docs

- The CI job is the test. Add a badge or a line in the README.
- Document the macOS build command in `CONTRIBUTING.md`.
- Note that the `scripts/cuda_env.sh` helper is Linux/WSL-specific.

### Risks

- GitHub's macOS runners are slower and may time out on the full test suite. Start with a
  reduced parity data set or cache the PyTorch install.
- Apple Clang may warn on C99 idioms that GCC accepts. Treat warnings as warnings, not
  errors, unless the repo already uses `-Werror` (it does not).
- Do not promise macOS CUDA support. There is no supported NVIDIA driver path.

---

# Optional polish

These are not required for 1.0. They improve onboarding and packaging.

## 13. [DONE] Finetune recipe

**Status:** not started  
**Effort:** 0.5 day for config/docs; 1-2 days for layer freezing

### Why

`--load-weights` already makes finetuning possible, but there is no documented recipe and no
example config. nanoGPT ships `config/finetune_shakespeare.py` with a constant LR and a
"save only on improvement" policy.

### Implementation

1. Add `configs/train/finetune.yaml`:

   ```yaml
   corpus: data/tinystories/train.bin
   load-weights: runs/pretrain/checkpoint.fblt
   steps: 2000
   optimizer: adamw
   lr: 0.00001
   lr-decay: 0
   weight-decay: 0.01
   eval-corpus: data/tinystories/heldout.bin
   eval-every: 100
   save-best: runs/finetune/best.fblt
   save-weights: runs/finetune/final.fblt
   ```

2. Document the workflow in `README.md`:

   - Start from a pretrained checkpoint.
   - Use a lower LR (typically 1/10 of pretraining).
   - Use a constant schedule or a short cosine decay.
   - Monitor held-out BPB; save the best checkpoint.

3. Optional layer freezing. Add `--freeze encoder|global|decoder` (repeatable) and a
   `freeze` mask in `args_t`. In `adamw_all`/`sgd_all`, skip parameters whose group is
   frozen. Gradients are still computed; to save time, skip the backward for a fully frozen
   module later. Start with the optimizer skip, which is simpler and correct.

   `blt_param_info` already exposes `is_enc`, `is_glob`, `is_dec`, so the predicate is easy.
   Add a test that frozen groups are bit-identical after a step.

### Acceptance

- Finetuning from a checkpoint lowers held-out BPB relative to the starting checkpoint.
- `--freeze encoder` leaves encoder weights bit-identical.
- The config is runnable with `fblt-train --config configs/train/finetune.yaml --backend ...`.

## 14. [DONE] Packaging extras

**Status:** not started  
**Effort:** 1 day

### Why

`pyproject.toml` declares only `pyyaml`. `bench_plots.py` needs `matplotlib` and `numpy`;
`prep_tinystories.py` needs `datasets`; parity generation needs `torch` and `numpy`.
`fblt/_binary.py` resolves `build/` relative to the current working directory, and CMake has
no `install()` rules, so `pip install fblt` cannot find its binaries.

### Implementation

1. Add metadata to `[project]`:

   ```toml
   readme = "README.md"
   license = {text = "Apache-2.0"}
   authors = [{name = "..."}]
   urls = {Homepage = "...", Repository = "https://github.com/iiEliJas/fBLT"}
   classifiers = [
     "License :: OSI Approved :: Apache Software License",
     "Programming Language :: C",
     "Programming Language :: Python :: 3",
     "Topic :: Scientific/Engineering :: Artificial Intelligence",
   ]
   ```

2. Add optional dependency groups:

   ```toml
   [project.optional-dependencies]
   dev = ["pytest", "ruff"]
   data = ["datasets"]
   plots = ["matplotlib", "numpy"]
   parity = ["torch", "numpy"]
   tracking = ["wandb"]
   ```

3. Make binary resolution robust in `fblt/_binary.py`:

   ```text
   1. FBLT_BIN_DIR environment variable
   2. <package_dir>/bin (if binaries are shipped as package data)
   3. build/ or build-cuda/ next to the repository root
   4. build/ or build-cuda/ in the current working directory (legacy)
   ```

   Resolve the package/repo root from `__file__`, not just CWD. Keep the current error
   message but list all searched paths.

4. Add CMake install rules if you want a system install:

   ```cmake
   install(TARGETS train_blt_d infer test_main
           RUNTIME DESTINATION bin)
   ```

   Decide whether 1.0 is a source-only release or a PyPI package. If source-only, say so
   explicitly in the README and do not imply `pip install fblt` is sufficient. If PyPI,
   ship wheels or a post-install build step; a Python package that requires a manual CMake
   build is confusing.

5. Add a `fblt doctor` command that prints the resolved binary paths, backend availability,
   Python version, and package version. This reduces support friction.

6. Add `python -m build` and `twine check dist/*` to a release workflow if publishing.

### Acceptance

- `pip install -e '.[dev,plots,data]'` installs everything needed for the common workflows.
- `fblt-train` and `fblt-infer` work from a directory other than the repo root when
  `FBLT_BIN_DIR` is set.
- The error message for a missing binary lists the paths searched.
- `python -m build` produces a wheel with the expected metadata.

## 15. Demo checkpoint

**Status:** not started  
**Effort:** 0.5 day + training time

### Why

The quickstart trains a 10-step debug model, which produces unusable text. The README's
sample output comes from a checkpoint that is not in the repo (and the README admits the
generation benchmark path no longer exists). New users cannot see a coherent sample without
a long training run. A small released checkpoint is the single biggest onboarding win.

### Implementation

1. Train a demo model on TinyStories with the published config (2.97M or 7.8M parameters).
   Save:
   - `model.fblt`
   - `entropy_lm.fblt` if entropy patching was used
   - `resolved_config.yaml`
   - `manifest.json` with SHA-256, byte size, config, corpus hash, git commit, and training
     command

2. Upload the files as GitHub Release assets, not into the git tree. The 7.8M FP32 model is
   roughly 31 MB; a release asset is appropriate.

3. Add `fblt/scripts/download_demo.py`:

   - `--out-dir` (default `runs/demo`)
   - `--url` or a pinned release tag
   - SHA-256 verification
   - refuses to overwrite unless `--force`

4. Add a README quickstart:

   ```sh
   python fblt/scripts/download_demo.py --out-dir runs/demo
   ./build/infer \
     --checkpoint runs/demo/model.fblt \
     --entropy-lm runs/demo/entropy_lm.fblt \
     --embed ... --hidden ... --enc-layers ... --glob-layers ... --dec-layers ... \
     --prompt "Once upon a time" --new-bytes 256 \
     --method blockdv --block-size 4 --adaptive --backend cpu
   ```

   Once item 10 lands, the shape flags can be omitted.

5. Add a CI check that downloads the demo with `--dry-run` or verifies the manifest
   checksum, so the link does not rot silently. A full download in CI may be too heavy;
   check the release asset with an HTTP HEAD and the manifest schema instead.

### Acceptance

- A fresh clone, build, and download produces coherent sample text without training.
- The manifest checksum matches the release asset.
- The README quickstart works on CPU and CUDA.
- The demo checkpoint is not committed to git.

## 16. Standard eval harness

**Status:** not started  
**Effort:** 3-5 days

### Why

fBLT has excellent internal diagnostics (`sanity_check`, `masked_acc`, `patch_*_split`,
`leak_probe`, etc.) and a corpus BPB eval, but no single command that evaluates a checkpoint
on a named benchmark corpus and writes a comparable JSON record. nanoGPT does not have this
either, so it is not a parity item; it is a "1.x research ergonomics" item.

### Implementation

1. Add `tools/eval_bpb.c` (or extend `sanity_check`) with:

   ```text
   --checkpoint FILE
   --embed E --hidden H --enc-layers N --glob-layers N --dec-layers N
   --corpus FILE
   --window W --windows N --skip S
   --patcher fixed|entropy --entropy-lm FILE
   --block-size B --layout paper|novel --t F
   --output-json FILE
   ```

   Output fields: `checkpoint`, `corpus`, `corpus_sha256`, `window`, `windows`, `patcher`,
   `causal_bpb`, `masked_bpb`, `masked_acc`, `params`, `git_commit`, `timestamp`.

2. Add a corpus prep script for one or two public text corpora (WikiText-103, enwik8) with
   SHA-256 checksums and a deterministic train/heldout split. TinyStories already has
   `prep_tinystories.py`; reuse its manifest format.

3. Add `bench/eval_suite.py` to run the tool over a matrix of checkpoints and corpora and
   aggregate the JSONL. Reuse `bench/harness.h`/`bench_report.py` where possible.

4. Add a CI smoke test on the sample corpus with a tiny checkpoint.

### Acceptance

- One command produces a JSON record with a reproducible BPB for a given checkpoint/corpus.
- The same checkpoint evaluates identically across CPU and CUDA within tolerance.
- The suite output can be plotted with the existing `bench_plots.py` or a small extension.

### Risks

- BPB is only comparable when the tokenizer/byte boundary and windowing match. Document the
  exact protocol and store it in the JSON.
- Entropy patching makes the eval stochastic unless the entropy LM and thresholds are
  fixed. Store the patcher config and seed.

---

# Release gates not covered by feature work

These are not features, but they block a credible 1.0. They are listed here so the release
checklist is in one place.

- [x] Bump `pyproject.toml` to `1.0.0` and fill `CHANGELOG.md`.
- [ ] Create a `v1.0.0` tag and a GitHub release.
- [x] Add PyPI metadata (`readme`, `license`, `authors`, `urls`, `classifiers`) or state
      explicitly that 1.0 is a source-only release.
- [x] Fix the stale docs paths:
      - `README.md` layout says `run/`, but the code is in `csrc/`.
      - `docs/CLI_REFERENCE.md` references `tools/run_*.sh`; no `tools/` directory exists.
      - `configs/ablations/` is missing, and `fblt/scripts/gen_configs.py` expects
        `configs/ablations/baseline_p4.json`.
      - `AGENTS.md` references `docs/Plan.md`; add `docs/PROFILING.md` and update the link.
      - `docs/CLI_REFERENCE.md` omits several built tools: `selfspec_audit`,
        `greedy_tf_acc`, `ownlatent_value`, `leak_probe`, `prefix_invariance`, `e2e-dv`,
        `bench-harness`.
- [ ] Add an end-to-end CI smoke test: tiny train -> save -> load -> infer. CI currently
      never runs `infer` or `e2e-dv`.
- [ ] Add a CUDA runtime CI runner or a documented manual validation matrix. CUDA CI is
      compile-only today.
- [ ] Decide the multi-GPU scope: implement (item 11) or state that 1.0 is single-GPU.
- [x] Add a consolidated "Known limitations" section: 128-patch cap, 7.8M demonstrated
      scale, CUDA non-determinism, greedy-only sampling for non-greedy methods.
- [ ] Publish the demo checkpoint (item 15).
- [x] Add a support matrix to the README (Linux CPU/CUDA, Windows CPU, macOS CPU, no macOS
      CUDA).

---

# Suggested execution order

## Week 1: cheap wins and release hygiene

1. Item 9 (selective weight decay) - stabilizes the optimizer before other training work.
2. Item 3 (warmup/cosine).
3. Item 1 (best checkpoint).
4. Item 4 (telemetry).
5. Item 5 (top-k/num-samples).
6. Item 6 (profiling docs/script).
7. Release metadata, docs path fixes, and the E2E CI smoke test.

## Week 2: observability and checkpoint portability

8. Item 2 (metrics JSONL + W&B).
9. Item 10 (checkpoint metadata v2).
10. Item 15 (demo checkpoint).
11. Item 14 (packaging extras) if publishing.

## 1.x: scale work

12. Item 7 (gradient accumulation).
13. Item 8 (BF16).
14. Item 11 (multi-GPU).
15. Item 12 (macOS) can land any time; it is independent.
16. Item 13 (finetune) and item 16 (eval harness) as capacity allows.

---

# Tracking checklist

## Before 1.0

- [x] 1. Best-checkpoint saving
- [x] 2. Metrics JSONL + optional W&B
- [x] 3. LR warmup + cosine/min-lr
- [x] 4. Training telemetry (params, throughput, MFU)
- [x] 5. `--num-samples` and `--top-k`
- [x] 6. Profiling target + docs

## 1.x

- [ ] 7. Gradient accumulation / batch > 1
- [ ] 8. BF16 mixed-precision training
- [x] 9. Selective weight decay
- [x] 10. Checkpoint metadata v2
- [ ] 11. Multi-GPU data parallel
- [x] 12. macOS support

## Optional

- [x] 13. Finetune recipe
- [x] 14. Packaging extras
- [ ] 15. Demo checkpoint
- [ ] 16. Standard eval harness
