"""Build results/*.csv from graded run outputs.

Inputs (paths are arguments so the script works on any machine):
  --grades      JSON mapping "<condition>|<question_id>" -> true/false (final grades after review)
  --questions   JSONL with one object per question (id, type), e.g. <work>/inputs/lme.jsonl
  --fc-graded   JSONL with id, source, cond, correct for FactConsolidation
  --fc-gold     fc_gold.jsonl: each FactConsolidation condition must cover exactly these question ids
  --c1-outputs  folder with the baseline answer files F/R/O.jsonl (prompt_tokens, reasoning_tokens, latency_s)
  --m1-calls    calls.jsonl from the mem0 runs (per-call tokens and latency)
  --m1-outputs  folder with the mem0 answer files M0.jsonl, Mnd.jsonl, ... (default: the folder of --m1-calls)
Question sets come from data/question_ids.json. Writes to --out (default: work/reproduced_results, so the
published results/ are not overwritten). Stops with an error, instead of writing tables, if any grade a table
needs is missing or a FactConsolidation condition does not cover its questions exactly once.

Answer telemetry (input tokens, reasoning tokens, latency) describes the answer call only: memory building,
retrieval and embedding are not included, and latency is the time of the answer API request. Only the call behind each graded answer (the last record per question)
counts: the current runner writes the same call_id into the answer record and the call log. Logs from the first
release of this repository tag the variant but have no call_id; there the graded answer's call is the first one
logged after it with the same answer length. The published logs tagged only the store, so the script matches
those calls to answers: for each question and store, calls and answers are taken in time order, and every pair
must agree on the answer length (the log records it) and on timing. Calls whose answer shares the same second
and the same length with another variant's answer are ambiguous and left out; telemetry_n gives the count used.

With the pipeline in this repo:
  python scripts/summarize_results.py --grades work/grades/lme_final.json --questions work/inputs/lme.jsonl --fc-graded work/grades/fc_graded.jsonl --fc-gold work/gold/fc_gold.jsonl --c1-outputs work/outputs --m1-calls work/outputs/calls.jsonl
"""
import argparse, collections, csv, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
for a in ["grades", "questions", "fc-graded", "c1-outputs", "m1-calls"]:
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--m1-outputs")
ap.add_argument("--fc-gold", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "..", "work", "reproduced_results"))
args = ap.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


final = json.load(open(args.grades))
qs = {r["id"]: r for r in load_jsonl(args.questions)}
ids = json.load(open(os.path.join(HERE, "..", "data", "question_ids.json"), encoding="utf-8"))
KU = sorted(ids["longmemeval_s_knowledge_update_76"])
REG = sorted(ids["longmemeval_s_regression_20"])
HIST = set(ids["knowledge_update_old_value_or_change_11"])     # ask about an old value or the direction of a change
BY_DAY, BY_TIME = "oldest to newest by day (same day: relevance order)", "oldest to newest by full timestamp"
COND = {  # condition -> (write-time dates, answer-time order, dates shown)
    "M0": ("no", "relevance", "no"), "Mnd": ("yes", "relevance", "no"), "Mord": ("yes", BY_DAY, "no"),
    "M": ("yes", BY_DAY, "yes"), "M_r2": ("yes", BY_DAY, "yes"), "M_r3": ("yes", BY_DAY, "yes"),
    "Mord_ts": ("yes", BY_TIME, "no"), "M_ts": ("yes", BY_TIME, "yes"),
    "M0_wo": ("no", "oldest to newest by write time (default store's created_at)", "no"),
    "F": ("n/a: full history", "", ""), "R": ("n/a: BM25 top-5 sessions", "", ""), "O": ("n/a: evidence sessions only", "", "")}
MEM0 = ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "Mord_ts", "M_ts", "M0_wo"]
KU_ROWS = ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "F", "R", "O"]
REG_ROWS = ["M0", "Mnd", "Mord", "M", "F", "R", "O"]
OPTIONAL = [c for c in ("M0_wo", "Mord_ts", "M_ts") if any(k.startswith(c + "|") for k in final)]   # newer conditions, if run

# ---------------------------------------------------------------- every grade a table needs must be there
need = [(c, q) for c in KU_ROWS + OPTIONAL for q in KU] + [(c, q) for c in REG_ROWS for q in REG]
absent = [f"{c}|{q}" for c, q in need if f"{c}|{q}" not in final]
if absent:
    sys.exit(f"{len(absent)} grades are missing (e.g. {absent[:5]}); finish the runs and grading first. "
             "Nothing was written.")
fc_rows = load_jsonl(args.fc_graded)
dup = [k for k, n in collections.Counter((r["cond"], r["id"]) for r in fc_rows).items() if n > 1]
if dup:
    sys.exit(f"fc-graded has duplicate rows, e.g. {dup[:3]}. Nothing was written.")
FC_SOURCES = ["factconsolidation_sh_6k", "factconsolidation_sh_32k"]
FC_CONDS = ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]
have = collections.defaultdict(set)
for r in fc_rows:
    have[(r["cond"], r["source"])].add(r["id"])
expect = collections.defaultdict(set)                         # the benchmark's question ids, not just those present
for g in load_jsonl(args.fc_gold):
    expect[g["source"]].add(g["id"])
gaps = [f"{c}/{s}: {len(expect[s] - have[(c, s)])} missing, {len(have[(c, s)] - expect[s])} unexpected"
        for c in FC_CONDS for s in FC_SOURCES if have[(c, s)] != expect[s]]
if gaps:
    sys.exit("FactConsolidation results are incomplete, so no scores are written:\n  " + "\n  ".join(gaps))

g = lambda c, q: bool(final[f"{c}|{q}"])
os.makedirs(args.out, exist_ok=True)

# per-question grades
with open(os.path.join(args.out, "longmemeval_per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question_id", "question_type", "asks_old_value_or_change", "condition", "correct"])
    for q in KU + REG:
        for c in COND:
            if f"{c}|{q}" in final:
                w.writerow([q, qs[q]["type"], q in HIST, c, int(g(c, q))])

# ---------------------------------------------------------------- answer-call telemetry, knowledge-update questions
tele = {}                                                     # condition -> (calls, source)
for c, files in [("F", ["F", "FKU"]), ("R", ["R", "RKU"]), ("O", ["O", "OKU"])]:
    last = {}                                                 # last record per id, as in grading
    for fn in files:
        p = os.path.join(args.c1_outputs, f"{fn}.jsonl")
        if os.path.exists(p):
            last.update({r["id"]: r for r in load_jsonl(p)})
    tele[c] = ([r for q, r in last.items() if q in KU and not r.get("error")], "answer records")

m1_out = args.m1_outputs or os.path.dirname(os.path.abspath(args.m1_calls))
answers = {c: load_jsonl(os.path.join(m1_out, f"{c}.jsonl")) if os.path.exists(os.path.join(m1_out, f"{c}.jsonl")) else []
           for c in MEM0}
graded = {(c, r["id"]): r for c in MEM0 for r in answers[c]}   # last record per (condition, id) is the graded one
latest_ts = {k: r["ts"] for k, r in graded.items()}
calls = [x for x in load_jsonl(args.m1_calls) if x["kind"] == "answer" and ":" in x["item"]]
groups, how, report = collections.defaultdict(list), collections.defaultdict(set), collections.Counter()
tagged = collections.defaultdict(list)
for x in calls:                                               # current runner: item = store:question:variant
    parts = x["item"].split(":")
    if len(parts) == 3 and parts[2] in MEM0 and parts[1] in KU:
        tagged[(parts[2], parts[1])].append(x)
for (c, q), xs in tagged.items():
    r = graded.get((c, q))
    if not r or r.get("error"):
        continue
    if r.get("call_id"):                                      # the call that produced the graded answer
        mine = [x for x in xs if x.get("call_id") == r["call_id"]][:1]
        source = "call id"
    else:                                                     # first-release logs: first matching call after it
        mine = [x for x in xs if x["ts"] >= r["ts"] and x["out_chars"] == len(r.get("answer") or "")][:1]
        source = "variant tag + answer length"
    if mine:
        groups[c].append(mine[0]); how[c].add(source)
    else:
        report["unmatched"] += 1
# published runs: item = store:question; store "lme0" = default store (M0, M0_wo), "lme"/"lme_M" = dated store
DEFAULT_STORE = ["M0", "M0_wo"]
STORE_CONDS = {"lme0": DEFAULT_STORE, "lme": [c for c in MEM0 if c not in DEFAULT_STORE],
               "lme_M": [c for c in MEM0 if c not in DEFAULT_STORE]}
untagged = collections.defaultdict(list)
for x in calls:
    store, q = x["item"].split(":")[:2]
    if len(x["item"].split(":")) == 2 and store in STORE_CONDS and q in KU:
        untagged[("lme0" if store == "lme0" else "lme", q)].append(x)
def align(xs, recs):
    """xs: calls in log order; recs: (ts, condition, record) of the same question and store. An answer record
    is written just before its call starts and the call is logged when it ends, so the k-th call belongs to
    the k-th record in time order. Records stamped in the same second form a block whose calls are told apart
    by answer length. Returns [(call, condition, ts)], the number of ambiguous calls, or None if anything
    disagrees."""
    blocks = [[r for r in recs if r[0] == ts] for ts in sorted({r[0] for r in recs})]
    if len(recs) != len(xs):
        return None
    out, ambiguous, i = [], 0, 0
    for b, block in enumerate(blocks):
        part, i = xs[i:i + len(block)], i + len(block)
        nxt = blocks[b + 1][0][0] if b + 1 < len(blocks) else None
        if any(x["ts"] < block[0][0] or (nxt and x["ts"] > nxt) for x in part):
            return None
        for x in part:
            same = [r for r in block if len(r[2].get("answer") or "") == x["out_chars"]]
            if not same:
                return None
            if len(same) > 1:
                ambiguous += 1
            else:
                out.append((x, same[0][1], same[0][0]))
    return out, ambiguous


for (store, q), xs in untagged.items():
    recs = [(r["ts"], c, r) for c in STORE_CONDS[store] for r in answers[c]
            if r["id"] == q and not r.get("error") and "run" not in r]
    res = align(xs, recs)
    if res is None:
        report["unmatched"] += len(xs)
        continue
    report["ambiguous"] += res[1]
    for x, c, ts in res[0]:
        if latest_ts.get((c, q)) == ts:                       # only the call behind the graded answer
            groups[c].append(x); how[c].add("matched from store-tagged log")
if report:
    print(f"telemetry: left out {report['ambiguous']} ambiguous calls; {report['unmatched']} graded answers or "
          "logged calls could not be matched")
for c in MEM0:
    tele[c] = (groups[c], " + ".join(sorted(how[c])) or "not available")


def stats(c):
    xs, src = tele.get(c, ([], "not available"))
    if not xs:
        return ["", "", "", 0, src]
    return [round(st.mean(x["prompt_tokens"] for x in xs)), round(st.median(x.get("reasoning_tokens") or 0 for x in xs)),
            round(st.median(x["latency_s"] for x in xs), 1), len(xs), src]


with open(os.path.join(args.out, "longmemeval_ku76_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "write_time_dates", "answer_order", "dates_shown", "correct", "n", "accuracy",
                "old_value_or_change_correct_of_11", "current_value_correct_of_65", "mean_answer_input_tokens",
                "median_answer_reasoning_tokens", "median_answer_api_latency_s", "telemetry_n", "telemetry_source"])
    for c in KU_ROWS[:6] + OPTIONAL + KU_ROWS[6:]:
        s = sum(g(c, q) for q in KU)
        w.writerow([c, *COND[c], s, len(KU), f"{s/len(KU):.3f}", sum(g(c, q) for q in KU if q in HIST),
                    sum(g(c, q) for q in KU if q not in HIST), *stats(c)])

with open(os.path.join(args.out, "longmemeval_decomposition_steps.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["step", "from", "to", "fixed", "broken", "net", "verdict"])
    steps = [("write-time dates", "M0", "Mnd"), ("chronological order (by day)", "Mnd", "Mord"),
             ("show dates", "Mord", "M"), ("whole intervention", "M0", "M")]
    if "M0_wo" in OPTIONAL:
        steps.append(("write order on the default store, no dates", "M0", "M0_wo"))
    if "Mord_ts" in OPTIONAL:
        steps.append(("full-timestamp order instead of by day", "Mord", "Mord_ts"))
    if "M_ts" in OPTIONAL:
        steps.append(("full-timestamp order instead of by day, dates shown", "M", "M_ts"))
    for label, a, b in steps:
        fixed = sum(1 for q in KU if not g(a, q) and g(b, q)); broken = sum(1 for q in KU if g(a, q) and not g(b, q))
        clear = fixed - broken >= 5 and fixed >= 3 * broken
        w.writerow([label, a, b, fixed, broken, fixed - broken, "clear effect" if clear else "no clear effect"])

with open(os.path.join(args.out, "longmemeval_regression20_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", "correct_of_20", "multi_session_of_10", "temporal_reasoning_of_10"])
    for c in REG_ROWS:
        w.writerow([c, sum(g(c, q) for q in REG), sum(g(c, q) for q in REG if qs[q]["type"] == "multi-session"),
                    sum(g(c, q) for q in REG if qs[q]["type"] == "temporal-reasoning")])

fc = collections.Counter()
for r in fc_rows:
    fc[(r["cond"], r["source"])] += bool(r["correct"])
n6, n32 = len(expect[FC_SOURCES[0]]), len(expect[FC_SOURCES[1]])
with open(os.path.join(args.out, "factconsolidation_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", f"sh_6k_correct_of_{n6}", f"sh_32k_correct_of_{n32}", f"total_of_{n6 + n32}"])
    for c in FC_CONDS:
        a, b = fc[(c, FC_SOURCES[0])], fc[(c, FC_SOURCES[1])]
        w.writerow([c, a, b, a + b])
with open(os.path.join(args.out, "factconsolidation_per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question_id", "source", "condition", "correct"])
    for r in fc_rows:
        w.writerow([r["id"], r["source"], r["cond"], int(r["correct"])])

print("wrote", sorted(os.listdir(args.out)))
