"""Export every graded answer, with the memories it was given (mem0 conditions) and any review note.

  python scripts/export_answers.py --grades lme_final.json --fc-graded fc_graded.jsonl \
      --lme-outputs DIR [DIR ...] --fc-outputs DIR --notes notes.json [--out DIR]

--lme-outputs  folders holding <condition>.jsonl answer files (M0, Mnd, Mord, M, M_r2, M_r3, F, R, O; the
               published baseline runs also used FKU/RKU/OKU file names for part of the questions)
--notes        JSON list of {"cond", "id", "note"} written during manual review (optional)
Writes answers_longmemeval.jsonl and answers_factconsolidation.jsonl into --out (default: work/reproduced_results).
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--grades", required=True)
ap.add_argument("--fc-graded", required=True)
ap.add_argument("--lme-outputs", nargs="+", required=True)
ap.add_argument("--fc-outputs", required=True)
ap.add_argument("--notes")
ap.add_argument("--out", default=os.path.join(HERE, "..", "work", "reproduced_results"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ids = json.load(open(os.path.join(HERE, "..", "data", "question_ids.json"), encoding="utf-8"))
use = ids["longmemeval_s_knowledge_update_76"] + ids["longmemeval_s_regression_20"]
final = json.load(open(args.grades))
notes = {(n["cond"], n["id"]): n["note"] for n in json.load(open(args.notes, encoding="utf-8"))} if args.notes else {}
FILES = {"M0": ["M0"], "Mnd": ["Mnd"], "Mord": ["Mord"], "M": ["M"], "M_r2": ["M_r2"], "M_r3": ["M_r3"],
         "F": ["F", "FKU"], "R": ["R", "RKU"], "O": ["O", "OKU"]}


def latest(path):
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


n = 0
with open(os.path.join(args.out, "answers_longmemeval.jsonl"), "w", encoding="utf-8") as f:
    for cond, names in FILES.items():
        rows = {}
        for d in args.lme_outputs:
            for name in names:
                rows.update(latest(os.path.join(d, f"{name}.jsonl")))
        for q in use:
            if q not in rows or f"{cond}|{q}" not in final:
                continue
            r = rows[q]
            rec = {"question_id": q, "condition": cond, "correct": bool(final[f"{cond}|{q}"]),
                   "review_note": notes.get((cond, q), ""), "answer": r.get("answer", "")}
            if "retrieved" in r:                           # memories handed to the answer model, in prompt order
                rec["memories"] = [(f"({m['date']}) " if r.get("dates_shown") or cond in ("M", "M_r2", "M_r3") else "")
                                   + m["text"] for m in r["retrieved"]]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
fc_ok = {(r["cond"], r["id"]): r["correct"] for r in map(json.loads, open(args.fc_graded, encoding="utf-8"))}
m = 0
with open(os.path.join(args.out, "answers_factconsolidation.jsonl"), "w", encoding="utf-8") as f:
    for cond in ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]:
        for q, r in latest(os.path.join(args.fc_outputs, f"fc_{cond}.jsonl")).items():
            rec = {"question_id": q, "source": r["source"], "condition": cond, "correct": fc_ok[(cond, q)],
                   "answer": r.get("answer", "")}
            if "retrieved" in r:
                rec["memories"] = [(f"(stored {x['order']}) {x['text']}" if isinstance(x, dict) else x) for x in r["retrieved"]]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); m += 1
print(f"wrote {n} LongMemEval answers and {m} FactConsolidation answers to {os.path.abspath(args.out)}")
