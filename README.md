# temporal-memory-eval

**Can an AI memory layer answer correctly after the user changes a fact?**

A user says *"I live in Chicago"*, and months later *"I moved to the suburbs"*. Then they ask *"Where do I live?"*. This repository is a controlled evaluation of how the open-source [mem0](https://github.com/mem0ai/mem0) memory layer handles such updates, what fixes them, and how much each part of the fix matters.

```
old fact  ──►  new fact  ──►  question about the current state
```

## Key findings

All numbers are from `results/*.csv`. Same answer model, same prompt template, same questions; only the memory pipeline changes.

1. **In this setup, default mem0 answered 22 of 76 update questions wrong (29%).** These are the 76 LongMemEval-S knowledge-update questions. Memories were listed as `- {memory}` in mem0's relevance order without dates, as in mem0's quick-start. This is a rate on this benchmark, not a general error rate.
2. **Keeping time fixes most of it.** Storing each conversation's real date and giving the answer model its memories in chronological order raises this to 68–70/76 (mean ≈ 91%). That is on par with feeding the whole ~105k-token chat history (68/76), at about 1k input tokens per question.
3. **Ordering is the step that matters.** Decomposing the intervention step by step:

   | Step | Fixed | Broken | Net | Verdict |
   |---|---|---|---|---|
   | Write-time dates (M0 → Mnd) | 10 | 5 | +5 | no clear effect |
   | **Chronological order + an "oldest to newest" note (Mnd → Mord)** | 11 | 2 | **+9** | **clear effect** |
   | Date labels instead of the note (Mord → M) | 2 | 2 | 0 | no clear effect |
   | Whole intervention (M0 → M) | 17 | 3 | +14 | clear effect |

   A step counts as a clear effect only if net ≥ 5 and fixed ≥ 3× broken. We wrote that rule down before the run. Two steps change more than one thing: Mnd → Mord adds both the ordering and a one-line "oldest to newest" note, and Mord → M drops that note while adding a date to each memory.

   Scope: on this memory store, written with real dates, explicit date labels added no net gain over the ordered list. For context, about 16% of the retrieved memory texts already contained a date written by the extractor. We did not test ordering on a store with no dates anywhere, or date labels without ordering.
4. **The same principle holds on a second benchmark.** On MemoryAgentBench FactConsolidation (200 conflicting-fact questions), mem0 scored 176/200. Two fixes reached 197 and 198:
   - labelling memories by write batch: 197/200;
   - re-ingesting with the extractor told to write one memory per fact, not to merge facts, and to keep each fact's serial number in the text: 198/200. This changes several things at once, so the gain cannot be attributed to the serial numbers alone.

   Feeding every fact directly scored 198/200.
5. **No aggregate regression observed in a 20-question check.** On 20 multi-session and temporal-reasoning questions, the whole intervention went from 13 to 16 correct (4 fixed, 1 broken, all on temporal-reasoning questions). Twenty questions can only reveal large regressions.
6. **Default imports write the wrong date.** When old conversations are imported, relative times like "last week" are resolved against *today*. The wrong absolute date then lands in the memory text. Of the 22 default-mode errors, 6 cite such dates, for example a cat adopted in early 2023 stored as "since around December 2025".

## Why it happens (mem0 2.1.0)

| Observation | Where |
|---|---|
| Extraction only adds memories; old and new versions of a fact coexist | `memory/main.py`, `_add_to_vector_store` |
| The extractor is asked to link contradicting memories (`linked_memory_ids`), but the link is not stored or used | `configs/prompts.py` vs `_add_to_vector_store` |
| OSS `add()` rejects `timestamp`; the extraction prompt's observation date defaults to today | `add()`, `generate_additive_extraction_prompt` |
| Search ranking combines semantic, BM25 and entity scores, with no recency term | `_search_vector_store`, `score_and_rank` |

Related upstream threads: [#4956](https://github.com/mem0ai/mem0/issues/4956) (stale facts after updates), [#4963](https://github.com/mem0ai/mem0/issues/4963) (import uses today's date), [#5352](https://github.com/mem0ai/mem0/issues/5352) (community workaround), [#4970](https://github.com/mem0ai/mem0/issues/4970) (`linked_memory_ids` ignored), [#6626](https://github.com/mem0ai/mem0/issues/6626) (recency in ranking).

## Reproduce the two mechanisms

- `repro/repro_ranking.py`: no LLM and no API key. It stores two dated memories and shows that search ranking ignores which one is newer. Recorded output: `repro/repro_ranking_output.txt`.
- `repro/repro_import_dates.py`: needs an LLM key. It imports a 2023 conversation and shows the extracted memory text dated to the run date. Recorded output: `repro/repro_import_dates_output.txt`.

## Setup of the main experiments

| | |
|---|---|
| Benchmarks | [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) and [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) FactConsolidation (single-hop, ~6k and ~32k tokens). Both are MIT-licensed and are not redistributed here; question IDs are in `data/question_ids.json` |
| Questions | 76 knowledge-update questions (all 78 except 2 used while developing the method), plus 20 multi-session and temporal-reasoning questions for regression |
| Memory layer | mem0 2.1.0, local Qdrant, BM25 hybrid search, top 20 memories, local `BAAI/bge-small-en-v1.5` embeddings. spaCy is not installed, so entity boosting is off, as in a default install |
| Models | DeepSeek V4.1 Flash (`deepseek-flash`). Answers use thinking mode; mem0's extraction runs without it |
| Baselines | Full chat history (F), BM25 top-5 sessions (R), gold evidence sessions only (O), same prompt template |

**Conditions on the 76 questions**

| Condition | Write-time dates | Answer order | Dates shown | Correct |
|---|---|---|---|---|
| M0, default | no | relevance | no | 54 |
| Mnd | yes | relevance | no | 59 |
| Mord | yes | chronological, labelled "oldest to newest" | no | 68 |
| M | yes | chronological | yes | 68 / 70 / 69 (three runs) |
| F / R / O | n/a | n/a | n/a | 68 / 67 / 72 |

"Write-time dates" means the session date is given to the extractor through mem0's `prompt` argument and stored as `created_at` metadata.

## How answers were graded

- **Automatic first pass**: substring and number matching.
- **Manual review**: every flagged or suspicious answer was re-judged with condition labels hidden and order shuffled. The review overturned 24 automatic "correct" verdicts. In each case the answer contained the gold value, but:
  - 13 listed the old and the new value without choosing one;
  - 9 gave the old value as the answer and mentioned the new one only in passing;
  - 2 had other errors: one reversed the direction of a change, one added an extra count.
- **Rules**:
  - Giving the latest value with a caveat that it may not be current counts as correct.
  - A correct answer that also mentions the old value counts as correct.
  - Reporting "the records conflict" without choosing counts as wrong.
- **Who graded**: grading was done mainly by an AI reviewer. The author spot-checked 22 items and overturned 2.
- **Dataset issues**: 2 LongMemEval gold answers appear inconsistent with their evidence. One (`852ce960`) is missed by every condition. The other (`a2f3aa27`, gold "1300" where the user only said "close to 1300") is graded correct only for the default condition M0, which answered "nearing 1300"; it therefore counts once in M0's favour (it is one of the 5 "broken" in the M0 → Mnd step). 2 FactConsolidation items are missed identically by FULL, MEM0_T and MEM0_S. All these items are kept in the scores.

## Limitations

- Synthetic conversations from two public benchmarks; no real users, so no claim about how often this happens in production.
- One model, one embedding model, one mem0 version (2.1.0).
- Most conditions ran once. Step verdicts come from per-question comparisons, not significance tests. Three repeated runs of M differed by up to 2 questions.
- Latency is a by-product, not a target: default-mode answers were slower (median 3.5 s vs 1.3–1.5 s), accompanied by about 3.5× more reasoning tokens (median 604 vs ~170). We observed the association; we did not establish that reasoning length explains the whole latency gap.
- In the logs of the published runs, token, reasoning and latency figures for Mnd, M and its reruns are pooled: those logs recorded the memory store, not the answer variant. The runner in `pipeline/` now tags every answer call with its variant, so fresh runs can be summarized per condition.
- The fix itself is not new. The contribution is measuring its effect under controlled conditions and isolating which part matters.

## Reproduce from scratch

Python 3.10. `pip install -r requirements.txt`. An API key for DeepSeek: the results use `deepseek-flash`, and any OpenAI-compatible endpoint can be set in the config.

1. **Download the data** (MIT-licensed, not included here):
   - `longmemeval_s_cleaned.json` and `longmemeval_oracle.json` from [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
   - `data/Conflict_Resolution-00000-of-00001.parquet` from [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
2. **Build the inputs.** This rebuilds exactly the prompts, sessions and gold answers we used: we checked all 96 questions byte for byte.
   ```
   python pipeline/prepare_data.py --longmemeval-s longmemeval_s_cleaned.json        --longmemeval-oracle longmemeval_oracle.json        --factconsolidation Conflict_Resolution-00000-of-00001.parquet
   ```
3. **Add your key.** Copy `pipeline/config.example.json` to `pipeline/config.local.json` (git-ignored) and fill in `api_key`.
4. **Run.** Add `--dry-run` to any command to see a cost estimate first. Re-run a command to resume; finished items are skipped.
   ```
   python pipeline/run_baselines.py                      # F / R / O
   python pipeline/run_mem0.py --task fc                 # FactConsolidation, all six conditions
   python pipeline/run_mem0.py --task lme                # 76 knowledge-update questions: M0, Mnd, Mord, M (+2 reruns)
   python pipeline/run_mem0.py --task lme --lme-types multi-session,temporal-reasoning --variants M,Mnd,Mord
   ```
5. **Grade.**
   - `python pipeline/grade.py auto` does the automatic pass and writes a blind review sheet.
   - Fill the `verdict` column of `work/grades/review_sheet.csv` with 1 (correct) or 0 (wrong).
   - Then run `python pipeline/grade.py apply`.
6. **Summarize.** Output goes to `work/reproduced_results/`, so the published `results/` are not overwritten.
   ```
   python scripts/summarize_results.py --grades work/grades/lme_final.json --questions work/inputs/lme.jsonl --fc-graded work/grades/fc_graded.jsonl --c1-outputs work/outputs --m1-calls work/outputs/calls.jsonl
   python scripts/export_answers.py --grades work/grades/lme_final.json --fc-graded work/grades/fc_graded.jsonl --lme-outputs work/outputs --fc-outputs work/outputs
   ```

**Cost and expected differences**
- **Cost**: roughly $12–13 at DeepSeek off-peak prices, about $25 at peak. This uses DeepSeek's prices as of September 2026. The estimates are conservative; mem0's write-time extraction is most of it. Embeddings run locally on CPU.
- **Budget check**: `budget_usd` ($20 by default) stops a single command when it is exceeded. It is not a cap on the total cost of a full reproduction, which runs several commands.
- **Answers vary**: they are sampled, so expect small differences. Three runs of M on the same memories ranged from 68 to 70.
- **Grading involves judgment**: the manual review is a human call. For comparison, `results/answers_*.jsonl` holds every answer we graded, with our final verdict and any review note. For mem0 conditions it also holds the memories the model was given. `results/*_per_question.csv` has the verdicts alone.

## Repository layout

```
pipeline/   prepare_data.py, run_baselines.py, run_mem0.py, grade.py, config.example.json
repro/      deterministic ranking repro + import-date repro, with recorded outputs
results/    summary and per-question CSVs, plus every graded answer (answers_*.jsonl)
scripts/    summarize_results.py and export_answers.py (build results/ from graded runs)
data/       question_ids.json (which benchmark questions were used)
```

## AI assistance

Experiment code, first-pass error analysis and grading drafts were produced with AI coding assistants (Claude, with reviews from ChatGPT and Codex). The author chose the questions, approved every pre-registered rule before each run, wrote predictions before the runs, spot-checked grading and made the scope decisions.

## Author

Xinyu Deng ([@empyreaaron](https://github.com/empyreaaron))

## License

Code in this repository: MIT (see `LICENSE`). Benchmark data belongs to its authors under their licenses.
