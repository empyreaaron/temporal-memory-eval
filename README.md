# temporal-memory-eval

**Can an AI memory layer answer correctly after the user changes a fact?**

A user says *"I live in Chicago"*, and months later *"I moved to the suburbs"*. Then they ask *"Where do I live?"*. This repository is a controlled evaluation of how the open-source [mem0](https://github.com/mem0ai/mem0) memory layer handles such updates, what fixes them, and how much each part of the fix matters.

```
old fact  ──►  new fact  ──►  question about the current state
```

## Key findings

All numbers are from `results/*.csv`. Every condition uses the same answer model, the same answer settings and the same questions. The prompts differ by condition, as listed under [Setup](#setup-of-the-main-experiments): the baselines put chat history in the prompt, the mem0 conditions put memories, and MEM0_T also uses a different conflict rule. Within each comparison below, only what the tables list changes.

1. **In this setup, default mem0 answered 22 of 76 update questions wrong (29%).** These are the 76 LongMemEval-S knowledge-update questions. Memories were listed as `- {memory}` in mem0's relevance order without dates, as in mem0's quick-start. This is a rate on this benchmark, not a general error rate.
2. **Keeping time fixes most of it.** Storing each conversation's real date and giving the answer model its memories oldest to newest raises this to 68–70/76 (mean ≈ 91%). That is on par with feeding the whole ~105k-token chat history (68/76), with about 1.1k tokens in the answer prompt. Both figures count the answer call only. Building the memory store is extra: in our runs it took a median of 48 extraction calls and about 527k input tokens per question's chat history (most of it cache hits). That cost is paid once per history and shared by every later question about it, which a single-question benchmark does not show.
3. **Ordering is the step that matters.** Decomposing the intervention step by step:

   | Step | Fixed | Broken | Net | Verdict |
   |---|---|---|---|---|
   | Write-time dates (M0 → Mnd) | 10 | 5 | +5 | no clear effect |
   | **Order by date + an "oldest to newest" note (Mnd → Mord)** | 11 | 2 | **+9** | **clear effect** |
   | Date labels instead of the note (Mord → M) | 2 | 2 | 0 | no clear effect |
   | Whole intervention (M0 → M) | 17 | 3 | +14 | clear effect |

   A step counts as a clear effect only if net ≥ 5 and fixed ≥ 3× broken. We wrote that rule down before the run. Two steps change more than one thing: Mnd → Mord adds both the ordering and a one-line "oldest to newest" note, and Mord → M drops that note while adding a date to each memory.

   The ordering is by calendar day. Memories from the same day kept mem0's relevance order, although the store holds their full timestamps. So "oldest to newest" was only true day by day: in 38 of the 76 questions, at least two same-day memories in the list were out of time order. This is a property of the list, not a count of wrong answers. The runner now also has full-timestamp variants (`Mord_ts`, `M_ts`), which have not been run.

   Scope: on this memory store, written with real dates, explicit date labels added no net gain over the ordered list. For context, about 16% of the retrieved memory texts already contained a date written by the extractor. We did not test ordering on a store with no dates anywhere, or date labels without ordering.
4. **The same principle holds on a second benchmark.** On MemoryAgentBench FactConsolidation (200 conflicting-fact questions), mem0 scored 176/200. Two fixes reached 197 and 198:
   - giving the answer model its memories in write order: sorted by mem0's `created_at`, each labelled with its write position, plus a rule that a larger label is newer: 197/200. Order, labels and rule changed together, so the 197 belongs to the combination, not to the labels alone. A label marks one distinct `created_at` value, not one `add()` call: the 10 add batches at 6k produced 17 distinct timestamps, and the 47 at 32k produced 84;
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

Related upstream threads: [#4956](https://github.com/mem0ai/mem0/issues/4956) (stale facts after updates), [#4963](https://github.com/mem0ai/mem0/issues/4963) (import uses today's date), [#5352](https://github.com/mem0ai/mem0/issues/5352) (community workaround), [#4970](https://github.com/mem0ai/mem0/issues/4970) (`linked_memory_ids` ignored), [#6626](https://github.com/mem0ai/mem0/pull/6626) (open PR: break exact score ties by recency).

## Reproduce the two mechanisms

- `repro/repro_ranking.py`: no LLM and no API key. It stores dated memories with `infer=False` and shows that search ranking ignores which one is newer. This tests ranking only, not mem0's extraction step. Scores are printed to 6 decimals: in the first case the older memory wins by 0.000026, which is not a tie. Recorded output (identical on mem0 2.1.0 and 2.2.1): `repro/repro_ranking_output.txt`.
- `repro/repro_import_dates.py`: needs an LLM key. It imports a 2023 conversation and shows the extracted memory text dated to the run date. The text is LLM-written, so a re-run gives different wording and a different date. Recorded output: `repro/repro_import_dates_output.txt`.

## Setup of the main experiments

| | |
|---|---|
| Benchmarks | [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval), cleaned release ([xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned); scores are not directly comparable with the original release), and [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) FactConsolidation (single-hop, ~6k and ~32k tokens). Both are MIT-licensed and are not redistributed here; question IDs are in `data/question_ids.json`, file checksums below |
| Questions | 76 knowledge-update questions (all 78 except 2 used while developing the method), plus 20 multi-session and temporal-reasoning questions for regression |
| Memory layer | mem0 2.1.0, local Qdrant, BM25 hybrid search, top 20 memories, local `BAAI/bge-small-en-v1.5` embeddings. spaCy is not installed, so entity boosting is off, as in a default install |
| Models | DeepSeek V4.1 Flash (`deepseek-flash`). Answers use thinking mode; mem0's extraction runs without it. The published runs recorded this requested alias, not the model version the API reported; the runners now record both |
| Baselines | Full chat history (F), BM25 top-5 sessions (R), gold evidence sessions only (O). The three share one chat-history prompt template |
| mem0 answer prompt | One memory prompt for all LongMemEval mem0 conditions; the conditions differ only in memory order, an "oldest to newest" line, and whether dates are shown (table below) |

**Conditions on the 76 questions**

| Condition | Write-time dates | Answer order | Dates shown | Correct |
|---|---|---|---|---|
| M0, default | no | relevance | no | 54 |
| Mnd | yes | relevance | no | 59 |
| Mord | yes | by day, labelled "oldest to newest" | no | 68 |
| M | yes | by day | yes | 68 / 70 / 69 (three runs) |
| F / R / O | n/a | n/a | n/a | 68 / 67 / 71 |

"Write-time dates" means the session date is given to the extractor through mem0's `prompt` argument and stored as `created_at` metadata. "By day" means sorted by calendar date: memories from the same day stay in mem0's relevance order (see finding 3).

**Data files used** (SHA-256):

| File | SHA-256 |
|---|---|
| `longmemeval_s_cleaned.json` | `35961662da991bec512124586e2e399a335e9e7c94272403e820eccc9946589e` |
| `longmemeval_oracle.json` | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` |
| `Conflict_Resolution-00000-of-00001.parquet` | `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45` |

## How answers were graded

- **Automatic first pass**: substring and number matching. FactConsolidation uses MemoryAgentBench's rule: correct if a normalized gold answer appears anywhere in the normalized answer. That rule would also count a gold "13" inside "130"; none of the 1,143 correct FactConsolidation verdicts depends on such a partial-token match.
- **Manual review**: every flagged or suspicious answer was re-judged with condition labels hidden and order shuffled. The review overturned 24 automatic "correct" verdicts. In each case the answer contained the gold value, but:
  - 13 listed the old and the new value without choosing one;
  - 9 gave the old value as the answer and mentioned the new one only in passing;
  - 2 had other errors: one reversed the direction of a change, one added an extra count.
- **Later re-check (2026-10-03)**: the automatic pass now also flags hedging or conflict wording ("does not confirm", "planned", "however", ...), a number that matches without the gold's words ("6 men" for "6 women"), and abstentions that go on to state facts. On the published answers this adds 14 answers to the review. One was overturned: O's answer to `07741c45` said the shoes were under the bed and the user later planned to move them, without saying where they are now. M_r2 and M_r3 gave the same kind of answer and were already graded wrong. O is now graded wrong too, so O drops from 72 to 71. No mem0 condition and no comparison between them changes.
- **Rules**:
  - Giving the latest value with a caveat that it may not be current counts as correct.
  - A correct answer that also mentions the old value counts as correct.
  - Reporting "the records conflict", or listing the old and new value without choosing, counts as wrong.
  - Reporting an old value plus a later plan to change it, without saying which holds now, counts as wrong.
- **Who graded**: grading was done mainly by an AI reviewer. The author spot-checked 22 items and overturned 2.
- **Dataset issues**: 2 LongMemEval gold answers appear inconsistent with their evidence. One (`852ce960`) is missed by every condition. The other (`a2f3aa27`, gold "1300" where the user only said "close to 1300") is graded correct only for the default condition M0, which answered "nearing 1300"; it therefore counts once in M0's favour (it is one of the 5 "broken" in the M0 → Mnd step). A third is debatable: `07741c45` takes the user's stated plan to move their sneakers to a closet shoe rack as done; every condition now misses it. The dataset is not consistent about plans: in `4d6b87c8` the gold counts only the confirmed value (25 titles, not the 27 the user planned to reach). So the same cautious answer, "the confirmed value is X; the planned change is not confirmed", is graded correct there and wrong here. We grade every question against its gold. 2 FactConsolidation items are missed identically by FULL, MEM0_T and MEM0_S. All these items are kept in the scores.

## Limitations

- Synthetic conversations from two public benchmarks; no real users, so no claim about how often this happens in production.
- One model, one embedding model, one mem0 version (2.1.0).
- Most conditions ran once. Step verdicts come from per-question comparisons, not significance tests. Three repeated runs of M differed by up to 2 questions.
- Token, reasoning and latency figures describe the answer call only. Latency is the time of the answer API request. It leaves out retrieval, embedding and building the memory store; for F/R/O in the published runs it also includes any retries.
- Latency is a by-product, not a target. Default-mode answers were slower (median 3.5 s vs 1.3–1.8 s for the dated store) and reasoned more: median 604 reasoning tokens for M0, 267 for Mnd, and 142–172 for the date-ordered conditions. We observed the association; we did not establish that reasoning length explains the whole latency gap.
- The published runs logged each answer call's memory store, not its answer variant. Per-condition figures were recovered by matching each logged call to its answer: same question and store, time order, and the exact answer length the log records. On the 76 questions, the four conditions that had been pooled (Mnd, M, M_r2, M_r3) made 304 calls, and all of them matched. 8 were left out as ambiguous (two variants answered in the same second with the same answer length), so some rows average 74 calls (`telemetry_n`). The same matching reproduces exactly the M0 and Mord figures, which were already separable (76 calls each). The runner now writes one call_id into each answer record and its call log entry, so only the call behind the graded answer counts.
- The fix itself is not new. The contribution is measuring its effect under controlled conditions and isolating which part matters.

## Reproduce from scratch

Python 3.10. `pip install -r requirements.txt`. An API key for DeepSeek: the results use `deepseek-flash`, and any OpenAI-compatible endpoint can be set in the config.

1. **Download the data** (MIT-licensed, not included here):
   - `longmemeval_s_cleaned.json` and `longmemeval_oracle.json` from [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
   - `data/Conflict_Resolution-00000-of-00001.parquet` from [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
2. **Build the inputs.** This rebuilds exactly the prompts, sessions and gold answers we used: we checked all 96 questions byte for byte.
   ```
   python pipeline/prepare_data.py --longmemeval-s longmemeval_s_cleaned.json --longmemeval-oracle longmemeval_oracle.json --factconsolidation Conflict_Resolution-00000-of-00001.parquet
   ```
   Check the downloads against the checksums above first; a different release gives different questions.
3. **Add your key.** Copy `pipeline/config.example.json` to `pipeline/config.local.json` (git-ignored) and fill in `api_key`.
4. **Run.** Add `--dry-run` to any command to see a cost estimate first. Re-run a command to resume: the last record per question decides, so finished items are skipped and failed ones retried. A command exits with an error code and prints `INCOMPLETE` until every planned answer exists; do not grade before it prints `done`. A key or balance error stops a command at once.
   ```
   python pipeline/run_baselines.py                      # F / R / O
   python pipeline/run_mem0.py --task fc                 # FactConsolidation, all six conditions
   python pipeline/run_mem0.py --task lme                # 76 knowledge-update questions: M0, Mnd, Mord, M (+2 reruns)
   python pipeline/run_mem0.py --task lme --lme-types multi-session,temporal-reasoning --variants M,Mnd,Mord
   ```
   Optional, not part of the published results: `--task lme --variants Mord_ts,M_ts --default-usage 0` orders memories by full timestamp instead of by day. Run it in the same work directory after the main run so it reuses the memory stores.

   Every answer record carries a fingerprint of the settings, prompts and inputs that produced it, and each memory store records how it was built. If you change the model, a prompt, `top_k` or the inputs, the runners refuse to resume over the old results instead of mixing them; use a new work directory (`TME_WORK`). Each run appends its settings (without the key), package versions and input checksums to `work/outputs/run_manifest.jsonl`.
5. **Grade.**
   - `python pipeline/grade.py auto` checks that every run is complete, does the automatic pass and writes a blind review sheet. It refuses to grade failed or missing answers (they are not wrong answers).
   - Fill the `verdict` column of `work/grades/review_sheet.csv` with 1 (correct) or 0 (wrong).
   - Then run `python pipeline/grade.py apply`. It checks that every row of the sheet has exactly one verdict and still matches the current answers.
6. **Summarize.** Output goes to `work/reproduced_results/`, so the published `results/` are not overwritten. The summary refuses to write tables if any grade is missing.
   ```
   python scripts/summarize_results.py --grades work/grades/lme_final.json --questions work/inputs/lme.jsonl --fc-graded work/grades/fc_graded.jsonl --fc-gold work/gold/fc_gold.jsonl --c1-outputs work/outputs --m1-calls work/outputs/calls.jsonl
   python scripts/export_answers.py --grades work/grades/lme_final.json --fc-graded work/grades/fc_graded.jsonl --lme-outputs work/outputs --fc-outputs work/outputs
   ```

Offline checks of this bookkeeping (no key, no network): `python -m unittest discover tests`.

**Cost and expected differences**
- **Cost**: roughly $12–13 at DeepSeek off-peak prices, about $25 at peak. This uses DeepSeek's prices as of September 2026. The estimates are conservative; mem0's write-time extraction is most of it. Embeddings run locally on CPU.
- **Budget check**: `budget_usd` ($20 by default) stops a single command once its spend passes the limit. The check runs before each API call, so calls already in flight can take the spend a little past it. It is not a cap on the total cost of a full reproduction, which runs several commands.
- **Answers vary**: they are sampled, so expect small differences. Three runs of M on the same memories ranged from 68 to 70.
- **Grading involves judgment**: the manual review is a human call. For comparison, `results/answers_*.jsonl` holds every answer we graded, with our final verdict and any review note. For mem0 conditions it also holds the memories the model was given. `results/*_per_question.csv` has the verdicts alone.

## Repository layout

```
pipeline/   prepare_data.py, run_baselines.py, run_mem0.py, grade.py, config.example.json
repro/      deterministic ranking repro + import-date repro, with recorded outputs
results/    summary and per-question CSVs, plus every graded answer (answers_*.jsonl)
scripts/    summarize_results.py and export_answers.py (build results/ from graded runs)
data/       question_ids.json (which benchmark questions were used)
tests/      offline checks of resuming, ordering, grading and summary bookkeeping
CHANGELOG.md  corrections made after the first release
```

## AI assistance

Experiment code, first-pass error analysis and grading drafts were produced with AI coding assistants (Claude, with reviews from ChatGPT and Codex). The author chose the questions, approved every pre-registered rule before each run, wrote predictions before the runs, spot-checked grading and made the scope decisions.

## Author

Xinyu Deng ([@empyreaaron](https://github.com/empyreaaron))

## License

Code in this repository: MIT (see `LICENSE`). Benchmark data belongs to its authors under their licenses.
