"""Export every graded answer, with the memories it was given (mem0 conditions) and any review note.

  python scripts/export_answers.py --grades lme_final.json --fc-graded fc_graded.jsonl \
      --lme-outputs DIR [DIR ...] --fc-outputs DIR --notes notes.json [--out DIR]

--lme-outputs  folders holding <condition>.jsonl answer files (M0, Mnd, Mord, M, M_r2, M_r3, Mord_ts, M_ts, M0_wo(_r2/_r3), F, R, O;
               the published baseline runs also used FKU/RKU/OKU file names for part of the questions)
--notes        JSON list of {"cond", "id", "note"} written during manual review (optional)
Writes answers_longmemeval.jsonl and answers_factconsolidation.jsonl into --out (default: work/reproduced_results).
The last record per question is the graded one (as in the runners and the grader). Grades and answers are
checked both ways before anything is written: the script stops, and leaves --out untouched, if a grade has no
answer, an answer has no grade, a graded answer's last record is an API error, or a grade appears twice.
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--grades", required=True)
ap.add_argument("--fc-graded", required=True)
ap.add_argument("--lme-outputs", nargs="+", required=True)
ap.add_argument("--fc-outputs", required=True)
ap.add_argument("--notes")
ap.add_argument("--out", default=os.path.join(HERE, "..", "work", "reproduced_results"))
args = ap.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}


ids = read_json(os.path.join(HERE, "..", "data", "question_ids.json"))
use = ids["longmemeval_s_knowledge_update_76"] + ids["longmemeval_s_regression_20"]
final = read_json(args.grades)
notes = {(n["cond"], n["id"]): n["note"] for n in read_json(args.notes)} if args.notes else {}
FILES = {"M0": ["M0"], "Mnd": ["Mnd"], "Mord": ["Mord"], "M": ["M"], "M_r2": ["M_r2"], "M_r3": ["M_r3"],
         "Mord_ts": ["Mord_ts"], "M_ts": ["M_ts"], "M0_wo": ["M0_wo"], "M0_wo_r2": ["M0_wo_r2"], "M0_wo_r3": ["M0_wo_r3"], "F": ["F", "FKU"], "R": ["R", "RKU"], "O": ["O", "OKU"]}
FC_CONDS = ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]
DATED = ("M", "M_r2", "M_r3")              # published records of these conditions predate the dates_shown field
problems = []

lme_out, answered = [], set()
for cond, names in FILES.items():
    rows = {}
    for d in args.lme_outputs:
        for name in names:
            rows.update(latest(os.path.join(d, f"{name}.jsonl")))
    for q in use:
        key, r = f"{cond}|{q}", rows.get(q)
        if r is None:
            continue
        answered.add(key)
        if r.get("error"):
            problems.append(f"LongMemEval {key}: the last record is an API error" + (" but it is graded" if key in final else ""))
        elif key not in final:
            problems.append(f"LongMemEval {key}: answer has no grade")
        else:
            rec = {"question_id": q, "condition": cond, "correct": bool(final[key]),
                   "review_note": notes.get((cond, q), ""), "answer": r.get("answer", "")}
            if "retrieved" in r:                           # memories handed to the answer model, in prompt order
                rec["memories"] = [(f"({m['date']}) " if r.get("dates_shown", cond in DATED) else "")
                                   + m["text"] for m in r["retrieved"]]
            lme_out.append(rec)
problems += [f"LongMemEval {k}: grade has no answer in --lme-outputs" for k in sorted(set(final) - answered)]

with open(args.fc_graded, encoding="utf-8") as f:
    fc_rows = [json.loads(l) for l in f if l.strip()]
counts = collections.Counter((r["cond"], r["id"]) for r in fc_rows)
problems += [f"FactConsolidation {c}|{q}: graded {n} times" for (c, q), n in counts.items() if n > 1]
fc_ok = {(r["cond"], r["id"]): r["correct"] for r in fc_rows}
fc_out, fc_answered = [], set()
for cond in FC_CONDS:
    for q, r in latest(os.path.join(args.fc_outputs, f"fc_{cond}.jsonl")).items():
        fc_answered.add((cond, q))
        if r.get("error"):
            problems.append(f"FactConsolidation {cond}|{q}: the last record is an API error"
                            + (" but it is graded" if (cond, q) in fc_ok else ""))
        elif (cond, q) not in fc_ok:
            problems.append(f"FactConsolidation {cond}|{q}: answer has no grade")
        else:
            rec = {"question_id": q, "source": r["source"], "condition": cond, "correct": fc_ok[(cond, q)],
                   "answer": r.get("answer", "")}
            if "retrieved" in r:
                rec["memories"] = [(f"(stored {x['order']}) {x['text']}" if isinstance(x, dict) else x) for x in r["retrieved"]]
            fc_out.append(rec)
problems += [f"FactConsolidation {c}|{q}: grade has no answer in --fc-outputs" for c, q in sorted(set(fc_ok) - fc_answered)]

if problems:
    sys.exit(f"Nothing was written: {len(problems)} grades and answers do not match. Finish the runs and regrade.\n  "
             + "\n  ".join(problems[:10]) + ("\n  ..." if len(problems) > 10 else ""))
os.makedirs(args.out, exist_ok=True)
for name, recs in [("answers_longmemeval.jsonl", lme_out), ("answers_factconsolidation.jsonl", fc_out)]:
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs)
print(f"wrote {len(lme_out)} LongMemEval answers and {len(fc_out)} FactConsolidation answers to {os.path.abspath(args.out)}")
