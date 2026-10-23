# Experiment Report Template

Copy this file to a new Markdown file in `docs/experiments/` and fill in the sections that
matter for the experiment. Delete sections that do not apply, but keep the distinction between
observation, interpretation, and conclusion.

File naming suggestion: `E###_short-topic.md`, using the experiment ID from
[EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md). Do not rename established reports only to fit a
naming scheme.

---

## Metadata

- **Experiment ID:** E###
- **Title:**
- **Status:** Planned / Exploratory / Checked / Independently reproduced / Superseded
- **Date or reporting period:**
- **Author or contributors:**
- **Related reports, issues, or papers:**
- **Git commit or release:**
- **Configurations and scripts:**
- **Artifacts:** checkpoints, result files, plots, or external locations with provenance.

## 1. Research question

State the question in one or two precise sentences.

## 2. Motivation and hypothesis

Explain what prompted the investigation and what outcome was expected, if a hypothesis was
defined. If the experiment is exploratory, say so.

## 3. Experimental design

Document:

- Compared methods or configurations.
- Baseline.
- Changed variables.
- Variables held constant.
- Datasets, models, and evaluation conditions.
- Metrics and measurement procedure.

## 4. Reproduction instructions

Include copyable commands, configuration paths, and artifact references. Record the
environment details required to reproduce the result, such as hardware, compiler, CUDA toolkit,
driver, and software versions.

If the exact historical environment is unavailable, state what is known and unknown.

## 5. Results

Present the actual measurements in a table or other clear format. Link to raw result files and
analysis scripts when available.

Avoid manually transcribing information that can be linked to an existing machine-readable
source.

## 6. Interpretation

Explain what the measurements suggest. Keep interpretation separate from observation and
identify plausible alternative explanations.

## 7. Limitations and validity threats

Document only the limitations relevant to this experiment, such as:

- small model scale
- limited dataset diversity
- noisy performance measurements
- nondeterminism
- missing baseline comparisons
- limited repetitions
- differences from published methods

## 8. Conclusion

State what the investigation established, what it did not establish, and how confidently the
result answers the initial question.

## 9. Follow-up work

Describe unanswered questions, promising extensions, and whether any conclusions or claims
elsewhere should be updated.
