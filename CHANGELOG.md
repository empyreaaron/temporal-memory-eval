# Changelog

## 2026-10-03: corrections after an independent review

The first release is commit `4c5807f`. An independent review of that release re-derived every published score from the raw run logs and found the counts correct. It also found the issues below. Apart from one reference score (O: 72 → 71), no score changed.

### Results

- **O on the 76 knowledge-update questions: 72 → 71.** Question `07741c45` asks where the user keeps their old sneakers now. O's answer said "under the bed" and that the user later planned to move them to a closet shoe rack, without saying which holds now. M_r2 and M_r3 gave the same kind of answer and were already graded wrong under the rule "listing the old and the new value without choosing is wrong". O is now graded the same way. The automatic pass had marked this answer correct and never sent it to review. The gold answer itself takes the user's plan as done, while the gold of `4d6b87c8` does not count a plan; both are listed under "Dataset issues". Grading stays against the gold, as for every other question; the author confirmed this choice.
- **Answer telemetry is now per condition.** Before, Mnd, M, M_r2 and M_r3 all showed one pooled figure (1044 tokens, 164 reasoning tokens, 1.3 s), because the published logs tagged the memory store, not the answer variant. That pool also missed the first 10 M answers, which were logged under a different tag (`lme_M:`). Each call is now matched to its answer by question, store, time order and exact answer length. The four pooled conditions made 304 calls on the 76 questions. All matched, and 8 ambiguous ones are left out; Mord's 76 calls on the same store were already separable. The matching reproduces the M0 and Mord figures that were already separate exactly. New values:

  | | Mnd | M | M_r2 | M_r3 |
  |---|---|---|---|---|
  | mean answer input tokens | 925 | 1085 | 1082 | 1082 |
  | median reasoning tokens | 267 | 154 | 145 | 142 |
  | median answer API latency (s) | 1.8 | 1.4 | 1.3 | 1.3 |
  | calls used | 74 | 74 | 74 | 74 |

- `results/longmemeval_ku76_summary.csv`: columns renamed to say what they measure (`mean_answer_input_tokens`, `median_answer_reasoning_tokens`, `median_answer_api_latency_s`), plus `telemetry_n` and `telemetry_source`. `answer_order` now says "by day" for Mord and M. In `longmemeval_decomposition_steps.csv` the step is now named "chronological order (by day)".

### Wording in the README

- Mord and M sort memories by calendar day, not by full timestamp. Memories from the same day keep mem0's relevance order, so in 38 of 76 questions some same-day memories were out of time order. The ordering result belongs to this day-level ordering.
- MEM0_T (197/200) changed memory order, write-order labels and the conflict rule together. Each label marks one distinct `created_at` value, not one `add()` batch (10 batches gave 17 labels at 6k, 47 gave 84 at 32k).
- "About 1k vs ~105k tokens" counts the answer call only. Building a question's memory store took a median of 48 extraction calls and about 527k input tokens, paid once per chat history.
- Latency is the answer API request time. It leaves out retrieval, embedding and memory building.
- The dataset is the cleaned LongMemEval-S release; SHA-256 checksums of all three data files are listed.
- The prompts differ by condition; the README now says which parts are shared.
- [#6626](https://github.com/mem0ai/mem0/pull/6626) is an open PR that breaks exact score ties by recency, not a general recency term.
- The ranking repro prints scores to 6 decimals. Its first case (0.307351 vs 0.307325) is not a tie.

### Pipeline (does not change how the published conditions are prompted)

- One resume rule everywhere: the last record per question decides. Before, the runners treated a question as finished if any earlier attempt had succeeded, while the grader read the last record.
- Answer records carry a fingerprint of the settings, prompts and inputs, plus a call_id that the call log repeats, so telemetry counts only the call behind each graded answer. Memory stores record how they were built. The runners refuse to resume over results made with other settings. Each run is logged in `outputs/run_manifest.jsonl`, and the model id reported by the API is saved.
- Runners exit with an error and print `INCOMPLETE` unless every planned answer exists. A key or balance error stops the baseline runner at once. A failed FactConsolidation ingest is recorded for each question it blocks.
- `grade.py auto` checks coverage and never grades a failed call as a wrong answer. `grade.py apply` refuses a review sheet with missing, duplicated or stale rows, and records whether each grade came from the automatic pass or the review. More answers go to review (hedging wording, number-only matches, abstentions that state facts). FactConsolidation now uses exactly the normalization of the published grader (same verdicts on all 1,200 published answers).
- `summarize_results.py` refuses to write tables when a grade is missing, and checks that each FactConsolidation condition covers exactly the question ids of the gold file (`--fc-gold` is now required). Before, a missing condition showed up as 0/200.
- `export_answers.py` checks grades and answers both ways before writing anything: a grade without an answer, an answer without a grade, a graded answer whose last record is an API error, or a duplicated grade stops the export.
- New optional variants `Mord_ts` and `M_ts` order memories by full timestamp. They have not been run.
- `tests/test_offline.py`: offline checks of the above.
