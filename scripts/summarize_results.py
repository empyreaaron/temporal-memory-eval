"""Build results/*.csv from graded run outputs.

Inputs (paths are arguments so the script works on any machine):
  --grades      JSON mapping "<condition>|<question_id>" -> true/false (final grades after review)
  --questions   JSONL with one object per question (id, type), e.g. <work>/inputs/lme.jsonl
  --fc-graded   JSONL with id, source, cond, correct for FactConsolidation
  --c1-outputs  folder with the baseline answer files F/R/O.jsonl (prompt_tokens, latency_s)
  --m1-calls    calls.jsonl from the mem0 runs (per-call tokens and latency)
Question sets come from data/question_ids.json. Writes to --out (default: work/reproduced_results, so the published results/ are not overwritten).

With the pipeline in this repo:
  python scripts/summarize_results.py --grades work/grades/lme_final.json --questions work/inputs/lme.jsonl       --fc-graded work/grades/fc_graded.jsonl --c1-outputs work/outputs --m1-calls work/outputs/calls.jsonl
"""
import argparse, collections, csv, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
for a in ["grades", "questions", "fc-graded", "c1-outputs", "m1-calls"]:
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "..", "work", "reproduced_results"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

final = json.load(open(args.grades))
qs = {r["id"]: r for r in map(json.loads, open(args.questions, encoding="utf-8"))}
ids = json.load(open(os.path.join(HERE, "..", "data", "question_ids.json"), encoding="utf-8"))
KU = sorted(ids["longmemeval_s_knowledge_update_76"])
REG = sorted(ids["longmemeval_s_regression_20"])
# knowledge-update questions that ask about an old value or the direction of a change
HIST = {"89941a94", "07741c44", "0977f2af", "10e09553", "50635ada", "9bbe84a2", "e66b632c",
        "6071bd76", "c4ea545c", "c6853660", "f685340e"}
COND = {  # condition -> (write-time dates, answer-time order, dates shown)
    "M0": ("no", "relevance", "no"), "Mnd": ("yes", "relevance", "no"), "Mord": ("yes", "chronological", "no"),
    "M": ("yes", "chronological", "yes"), "M_r2": ("yes", "chronological", "yes"), "M_r3": ("yes", "chronological", "yes"),
    "F": ("n/a: full history", "", ""), "R": ("n/a: BM25 top-5 sessions", "", ""), "O": ("n/a: evidence sessions only", "", "")}
g = lambda c, q: bool(final[f"{c}|{q}"])

# per-question grades
with open(os.path.join(args.out, "longmemeval_per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question_id", "question_type", "asks_old_value_or_change", "condition", "correct"])
    for q in KU + REG:
        for c in COND:
            if f"{c}|{q}" in final:
                w.writerow([q, qs[q]["type"], q in HIST, c, int(g(c, q))])

# token / latency per condition (knowledge-update questions)
tok, lat, think = {}, {}, {}
for c, files in [("F", ["F", "FKU"]), ("R", ["R", "RKU"]), ("O", ["O", "OKU"])]:
    rows = [r for fn in files if os.path.exists(os.path.join(args.c1_outputs, f"{fn}.jsonl"))
            for r in map(json.loads, open(os.path.join(args.c1_outputs, f"{fn}.jsonl"), encoding="utf-8"))
            if r["id"] in KU and not r.get("error")]
    tok[c] = round(st.mean(r["prompt_tokens"] for r in rows)); lat[c] = round(st.median(r["latency_s"] for r in rows), 1)
    think[c] = round(st.median(r.get("reasoning_tokens") or 0 for r in rows))
calls = [x for x in map(json.loads, open(args.m1_calls, encoding="utf-8"))
         if x["kind"] == "answer" and ":" in x["item"] and x["item"].split(":")[1] in KU]
if any(len(x["item"].split(":")) == 3 for x in calls):      # current runner: item = store:question:variant
    groups = {c: [x for x in calls if x["item"].split(":")[-1] == c] for c in ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3"]}
    pick = lambda c: groups[c]
else:   # logs from the published runs tag only the store, so group by store and run date (some figures pooled):
        #   default store (lme0) -> M0; patched store (lme) on 2026-10-01 -> Mord; before -> M/Mnd/M_r2/M_r3
    groups = {"M0": [x for x in calls if x["item"].startswith("lme0:")],
              "Mord": [x for x in calls if x["item"].startswith("lme:") and x["ts"] >= "2026-10-01"],
              "M": [x for x in calls if x["item"].startswith("lme:") and x["ts"] < "2026-10-01"]}
    pick = lambda c: groups["M0" if c == "M0" else "Mord" if c == "Mord" else "M"]
for c in ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3"]:
    xs = pick(c)
    tok[c] = round(st.mean(x["prompt_tokens"] for x in xs)); lat[c] = round(st.median(x["latency_s"] for x in xs), 1)
    think[c] = round(st.median(x.get("reasoning_tokens") or 0 for x in xs))

with open(os.path.join(args.out, "longmemeval_ku76_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "write_time_dates", "answer_order", "dates_shown", "correct", "n", "accuracy",
                "old_value_or_change_correct_of_11", "current_value_correct_of_65", "mean_input_tokens", "median_reasoning_tokens", "median_latency_s"])
    for c in ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "F", "R", "O"]:
        s = sum(g(c, q) for q in KU)
        w.writerow([c, *COND[c], s, len(KU), f"{s/len(KU):.3f}", sum(g(c, q) for q in KU if q in HIST),
                    sum(g(c, q) for q in KU if q not in HIST), tok[c], think[c], lat[c]])

with open(os.path.join(args.out, "longmemeval_decomposition_steps.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["step", "from", "to", "fixed", "broken", "net", "verdict"])
    for label, a, b in [("write-time dates", "M0", "Mnd"), ("chronological order", "Mnd", "Mord"),
                        ("show dates", "Mord", "M"), ("whole intervention", "M0", "M")]:
        fixed = sum(1 for q in KU if not g(a, q) and g(b, q)); broken = sum(1 for q in KU if g(a, q) and not g(b, q))
        clear = fixed - broken >= 5 and fixed >= 3 * broken
        w.writerow([label, a, b, fixed, broken, fixed - broken, "clear effect" if clear else "no clear effect"])

with open(os.path.join(args.out, "longmemeval_regression20_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", "correct_of_20", "multi_session_of_10", "temporal_reasoning_of_10"])
    for c in ["M0", "Mnd", "Mord", "M", "F", "R", "O"]:
        w.writerow([c, sum(g(c, q) for q in REG), sum(g(c, q) for q in REG if qs[q]["type"] == "multi-session"),
                    sum(g(c, q) for q in REG if qs[q]["type"] == "temporal-reasoning")])

fc = collections.defaultdict(lambda: [0, 0])
fc_rows = [json.loads(l) for l in open(args.fc_graded, encoding="utf-8")]
for r in fc_rows:
    fc[(r["cond"], r["source"])][0] += r["correct"]; fc[(r["cond"], r["source"])][1] += 1
with open(os.path.join(args.out, "factconsolidation_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", "sh_6k_correct_of_100", "sh_32k_correct_of_100", "total_of_200"])
    for c in ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]:
        a, b = fc[(c, "factconsolidation_sh_6k")][0], fc[(c, "factconsolidation_sh_32k")][0]
        w.writerow([c, a, b, a + b])
with open(os.path.join(args.out, "factconsolidation_per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question_id", "source", "condition", "correct"])
    for r in fc_rows:
        w.writerow([r["id"], r["source"], r["cond"], int(r["correct"])])

print("wrote", sorted(os.listdir(args.out)))
