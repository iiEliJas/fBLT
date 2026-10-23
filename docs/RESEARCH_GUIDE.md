# fBLT Research Guide

This guide explains how to investigate questions about fBLT in a way that is reproducible,
traceable, and honest about the strength of the evidence. It is written for the actual
repository and its existing tools; it is not a generic research-methodology essay.

For the complete command and flag reference, see [CLI_REFERENCE.md](CLI_REFERENCE.md).
For the end-to-end user workflow, see [STARTER_GUIDE.md](STARTER_GUIDE.md).

## 1. Research goals and scope

Working on fBLT can produce several different kinds of outcome. They are related, but they
should not be confused:

1. **Implementing published research.** Reproducing the BLT or Fast-BLT architecture and
   training procedure in this codebase.
2. **Verifying implementation correctness.** Finding deviations between the code and the
   papers, then fixing them.
3. **Reproducing published findings.** Running a comparable experiment and checking whether
   the reported trend survives.
4. **Investigating behavior under new conditions.** Testing a method at a different scale,
   dataset, schedule, or hardware configuration.
5. **Producing a genuinely new finding.** Establishing something not already claimed by the
   papers or prior reports.

fBLT is an independent implementation with experimentally demonstrated limits. The largest
demonstrated model is 7.8M parameters on TinyStories. A result at that scale does not
automatically establish a claim about larger models, other datasets, or published systems.
Describe exactly what was tested, and avoid generalizing beyond the available evidence.

Not every code change is a scientific contribution, and not every successful experiment
establishes a general claim. A bug fix, performance improvement, or documentation change
can be valuable without becoming a research finding.

## 2. Formulating a research question

A useful experiment starts with a question that can be answered by a measurement. Before
running anything, record:

- **Research question:** one or two precise sentences.
- **Motivation:** why the question matters and what prior finding it relates to.
- **Hypothesis:** what outcome is expected, if a hypothesis exists. Exploratory work may
  have no hypothesis.
- **Variable investigated:** the single change or comparison under study.
- **Baseline or comparison:** the configuration or method used for comparison.
- **Controlled conditions:** model, data, schedule, seeds, hardware, and other settings that
  should stay fixed.
- **Measurements:** the metrics that can answer the question.
- **Limitations and alternatives:** what else could explain the result.

### Exploratory versus hypothesis-testing experiments

An **exploratory** experiment observes behavior without a prior expectation. It is appropriate
when the phenomenon is unknown, when debugging, or when checking whether a measurement is
worth pursuing. Its conclusions should be labeled exploratory.

A **hypothesis-testing** experiment changes one variable and compares against a baseline.
It should define the comparison before seeing the result. If the hypothesis changes after the
data is seen, report that honestly.

### fBLT-specific examples

These are illustrative questions, not instructions to run new experiments:

- Does entropy-based patching improve BLT-DV acceptance over fixed-stride patching at the
  same trained checkpoint?
- How does training block size affect inference acceptance and output agreement?
- Does a BLT-DV advantage persist at a larger model scale or on a different corpus?
- Is a CUDA kernel speedup still visible in end-to-end generation latency?

Do not present these hypotheses as established facts.

## 3. Designing fair comparisons

Comparisons are interpretable only when the differences are deliberate and documented.

- Use comparable model dimensions, datasets, schedules, and evaluation procedures.
- Change one variable at a time unless the experiment is explicitly designed to test a
  combination.
- Record every difference between compared methods, including defaults that are easy to miss.
- Keep training and evaluation data disjoint. Check for overlap when possible.
- Record random seeds and deterministic settings. CUDA results are non-deterministic unless
  `--deterministic` is passed.
- Repeat measurements when timing noise or small sample sizes could affect the conclusion.
- Distinguish a single observation from a reliable trend.
- Choose metrics that address the question. BPB, masked accuracy, acceptance, agreement,
  NFE/byte, bandwidth, and wall-clock answer different questions.
- Avoid comparing results from different model scales, datasets, or hardware without stating
  that limitation.

Not every experiment needs multiple training runs, and not every measurement needs statistical
analysis. The methodology should match the question and the likely sources of uncertainty.
Do not impose a universal number of seeds or repetitions without justification.

## 4. Reproducibility checklist

Use this checklist for experiments that support important claims. Irrelevant fields may be
omitted; for example, an isolated kernel benchmark may not need a training dataset.

- [ ] Git commit, release tag, or other exact source version.
- [ ] Commands and effective configuration, including `resolved_config.yaml` when available.
- [ ] Model checkpoint and associated artifacts, with names and provenance.
- [ ] Dataset identity, version, provenance, and preprocessing.
- [ ] Evaluation split and any data-leakage checks.
- [ ] Model dimensions and training parameters.
- [ ] Inference settings, including block size and patching configuration.
- [ ] Random seeds and deterministic settings.
- [ ] Hardware, GPU model, driver, CUDA toolkit, compiler, and software versions.
- [ ] Measurement procedure and number of repeated runs.
- [ ] Raw measurements and analysis scripts.
- [ ] Known deviations from the intended protocol.
- [ ] Limitations and remaining uncertainty.

For large artifacts, do not commit checkpoints or datasets merely to make a report look
complete. Reference release assets or documented external locations instead.

## 5. Interpreting results

Keep these categories separate:

- **Observed result:** what the measurement says.
- **Interpretation:** what the experimenter thinks it means.
- **Hypothesis:** a proposed mechanism.
- **Conclusion:** what the experiment actually establishes.
- **Unresolved question:** what remains unknown.

A performance improvement that correlates with a change does not establish the mechanism
causing that improvement. Correlation, implementation details, or measurement procedure may
explain the result. State alternative explanations when they are plausible.

Negative and inconclusive results are valuable. Report them rather than silently dropping
them. A failed experiment can prevent another contributor from repeating the same work.

CUDA nondeterminism and noisy timing measurements can affect conclusions. When a result
depends on timing, report the hardware, measurement conditions, and whether runs were
repeated.

## 6. Performance measurement principles

fBLT has several distinct measurement layers. Do not mix them without explanation:

1. **Kernel-level microbenchmarks:** isolated operation throughput.
2. **Training throughput:** steps, bytes, or tokens processed per second.
3. **End-to-end inference benchmarks:** generation latency, NFE/byte, bandwidth, and
   acceptance.
4. **Model-quality evaluations:** BPB, masked accuracy, agreement, and task metrics.

A speedup is meaningful only when the measured operations, configurations, and conditions
are sufficiently comparable. Report the actual metric, baseline, hardware, and measurement
conditions. A kernel speedup does not imply an end-to-end speedup, and a single-run timing
is not a trend.

Preserve the project's existing benchmark methodology and link to
[BENCHMARKS.md](BENCHMARKS.md) for authoritative implementation-specific instructions.
Do not duplicate the complete benchmarking manual here.

## 7. Reporting findings

A useful experiment report should let a reader answer:

- What question was asked?
- Why did it matter?
- What exactly was done?
- What happened?
- How strong is the evidence?
- What does the result imply?
- What are its limitations?
- How could somebody reproduce or extend it?

Use [docs/experiments/TEMPLATE.md](experiments/TEMPLATE.md) for new reports. Keep reports as
short as the experiment allows; a simple afternoon experiment does not need a long report.

## 8. Contributing experiments

A lightweight workflow for external contributors:

1. Identify a question or an open experiment in [todo.md](../todo.md).
2. Check [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) to see whether it has already been
   investigated.
3. Open an issue or discussion when coordination would be useful.
4. Define the baseline and intended measurements before running.
5. Run the experiment and preserve relevant evidence.
6. Write up the findings, including negative results.
7. Submit the report or associated changes for review.

Issue creation is not mandatory for every tiny exploratory run. It is valuable when work
might overlap, when a result may change a documented claim, or when the experiment requires
coordination.

Link to [CONTRIBUTING.md](../CONTRIBUTING.md) for code and documentation contribution rules.

## 9. Experiment log and statuses

The central index is [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). It is a compact index of
significant research activity, not a chronological dump of every command or measurement.

Statuses:

- **Planned:** defined, but not yet run.
- **Exploratory:** initial observations exist, but the result has not been sufficiently
  checked.
- **Checked:** the setup and results have been examined and the report contains the available
  evidence. This does not imply independent reproduction.
- **Independently reproduced:** a separate rerun has confirmed the reported result within
  the documented conditions and tolerances.
- **Superseded:** a later experiment or correction has replaced the result as the current
  reference.

These are evidence and workflow statuses, not quality rankings. A report can be useful when
its result is exploratory or inconclusive. Never mark an experiment independently
reproduced unless evidence supports that designation.

## 10. Evidence, data, and artifact policies

- Prefer machine-readable result files and reproducible analysis scripts over manual
  transcription when practical.
- Use the repository's existing formats and tools. Do not introduce new dependencies or
  services.
- Do not commit large checkpoints, private datasets, or generated artifacts merely to make
  a report look complete.
- Reference release assets or documented external locations, and record artifact names and
  provenance.
- Never fabricate an artifact URL or imply that an external resource is archived if it is not.
- Document dataset origin, licensing, preprocessing, and held-out evaluation procedures when
  relevant.
- Protect evaluation integrity by checking for overlap between training and evaluation data
  where feasible.
- Retain enough history to explain invalidated results. The corrected result should be clearly
  identified as current, with a link to the earlier report when relevant.

## 11. fBLT-specific considerations

- **Kernel versus end-to-end:** a fast kernel or matmul does not imply a faster model or
  generation path. Measure at the level the claim concerns.
- **Model scale:** results at 7.8M parameters may not transfer to larger models. State the
  scale and avoid broad generalization.
- **Entropy and patching compatibility:** block-diffusion inference must use the same entropy
  LM and block size as training. A mismatch can make a result meaningless.
- **Training versus evaluation data:** evaluation must use held-out bytes. Training and eval
  corpora must not overlap.
- **Block size:** `B` must match training. Results at `B=8` for a model trained at `B=4`
  measure a train/inference mismatch, not a capability limit.
- **CUDA nondeterminism:** atomic reductions and cuBLAS can introduce run-to-run variance.
  Use `--deterministic` for ablations when reproducibility matters.
- **Implementation versus publication:** a code change that fixes a deviation from a paper is
  an implementation correction. A reproduction or new finding requires evidence beyond the
  code change.
- **Benchmark versus quality:** acceptance rates, NFE/byte, bandwidth, BPB, and generation
  quality answer different questions. Do not substitute one for another.

## 12. Scope and maintenance

This workflow is intentionally lightweight. It does not require a database, tracker, automated
experiment runner, or new dependency. It should remain easy for one maintainer to keep
current.

Not every debugging run needs a formal report. Record significant experiments, benchmark
findings, negative results worth sharing, and investigations referenced elsewhere.
