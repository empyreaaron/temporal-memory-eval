"""Build every model input and gold file from the public benchmarks.

Download the data yourself (both MIT-licensed), then run from the repository root:
    python pipeline/prepare_data.py \
        --longmemeval-s      path/to/longmemeval_s_cleaned.json \
        --longmemeval-oracle path/to/longmemeval_oracle.json \
        --factconsolidation  path/to/Conflict_Resolution-00000-of-00001.parquet

LongMemEval:       https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
MemoryAgentBench:  https://huggingface.co/datasets/ai-hyz/MemoryAgentBench  (data/Conflict_Resolution-*.parquet)

Questions come from data/question_ids.json (76 knowledge-update + 20 regression questions).
Writes into the work directory (default ./work, or set TME_WORK):
  inputs/lme.jsonl        sessions per question, for the mem0 runs (no answers)
  inputs/F|R|O.jsonl      prompts for the baselines: full history, BM25 top-5 sessions, evidence sessions
  inputs/fc.jsonl         FactConsolidation facts and questions (no answers)
  gold/lme_gold.jsonl, gold/fc_gold.jsonl   answers, read only by the graders
"""
import argparse, collections, json, math, os, re

PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PIPE)
WORK = os.environ.get("TME_WORK", os.path.join(ROOT, "work"))
FC_SOURCES = ["factconsolidation_sh_6k", "factconsolidation_sh_32k"]

TEMPLATE = ("I will give you several history chats between you and a user, each with its date. "
            "Please answer the question based on the relevant chat history. "
            "If the history does not contain enough information, say so.\n\n"
            "History Chats:\n\n{history}\n\nCurrent Date: {date}\nQuestion: {question}\nAnswer (concise):")

tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())


def bm25_rank(query, docs, k1=1.5, b=0.75):
    dt = [tok(d) for d in docs]
    n, avg = len(dt), sum(map(len, dt)) / len(dt)
    df = collections.Counter(w for d in dt for w in set(d))
    scores = []
    for d in dt:
        tf, s = collections.Counter(d), 0.0
        for w in tok(query):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])


def sess_text(sess):
    return "\n".join(f"{t['role']}: {t['content']}" for t in sess)


def history(pairs):                                   # (date, session) pairs, shown in date order
    pairs = sorted(pairs, key=lambda p: p[0])
    return "\n\n".join(f"### Session {k}\nSession Date: {d}\nSession Content:\n{sess_text(s)}"
                       for k, (d, s) in enumerate(pairs, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longmemeval-s", required=True)
    ap.add_argument("--longmemeval-oracle", required=True)
    ap.add_argument("--factconsolidation", required=True)
    args = ap.parse_args()
    ids = json.load(open(os.path.join(ROOT, "data", "question_ids.json"), encoding="utf-8"))
    use = ids["longmemeval_s_knowledge_update_76"] + ids["longmemeval_s_regression_20"]
    os.makedirs(os.path.join(WORK, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "gold"), exist_ok=True)

    S = {x["question_id"]: x for x in json.load(open(args.longmemeval_s, encoding="utf-8"))}
    O = {x["question_id"]: x for x in json.load(open(args.longmemeval_oracle, encoding="utf-8"))}
    missing = [q for q in use if q not in S or q not in O]
    if missing:
        raise SystemExit(f"{len(missing)} question ids not found in the LongMemEval files, e.g. {missing[:3]}")
    files = {c: open(os.path.join(WORK, "inputs", f"{c}.jsonl"), "w", encoding="utf-8") for c in "FRO"}
    with open(os.path.join(WORK, "inputs", "lme.jsonl"), "w", encoding="utf-8") as f, \
         open(os.path.join(WORK, "gold", "lme_gold.jsonl"), "w", encoding="utf-8") as g:
        for qid in use:
            s, o = S[qid], O[qid]
            sessions = sorted(
                ({"date": d, "session_id": sid, "turns": [{"role": t["role"], "content": t["content"]} for t in sess]}
                 for d, sid, sess in zip(s["haystack_dates"], s["haystack_session_ids"], s["haystack_sessions"])),
                key=lambda x: x["date"])
            f.write(json.dumps({"id": qid, "type": s["question_type"], "question": s["question"],
                                "question_date": s["question_date"], "sessions": sessions}, ensure_ascii=False) + "\n")
            g.write(json.dumps({"id": qid, "type": s["question_type"], "abstention": qid.endswith("_abs"),
                                "question": s["question"], "question_date": s["question_date"], "answer": s["answer"],
                                "answer_session_ids": s["answer_session_ids"]}, ensure_ascii=False) + "\n")
            full = list(zip(s["haystack_dates"], s["haystack_sessions"]))
            order = bm25_rank(s["question"], [sess_text(x) for x in s["haystack_sessions"]])
            ctx = {"F": full, "R": [full[i] for i in order[:5]],
                   "O": list(zip(o["haystack_dates"], o["haystack_sessions"]))}
            for c, pairs in ctx.items():
                prompt = TEMPLATE.format(history=history(pairs), date=s["question_date"], question=s["question"])
                files[c].write(json.dumps({"id": qid, "prompt": prompt}, ensure_ascii=False) + "\n")
    for fh in files.values():
        fh.close()
    print(f"LongMemEval: {len(use)} questions -> inputs/lme.jsonl, inputs/F|R|O.jsonl, gold/lme_gold.jsonl")

    import pyarrow.parquet as pq
    rows = {r["metadata"]["source"]: r for r in pq.read_table(args.factconsolidation).to_pylist()}
    with open(os.path.join(WORK, "inputs", "fc.jsonl"), "w", encoding="utf-8") as f, \
         open(os.path.join(WORK, "gold", "fc_gold.jsonl"), "w", encoding="utf-8") as g:
        for src in FC_SOURCES:
            r = rows[src]
            facts = [l for l in r["context"].split("\n") if re.match(r"^\d+\. ", l)]
            qids = r["metadata"]["qa_pair_ids"]
            f.write(json.dumps({"source": src, "facts": facts,
                                "questions": [{"id": i, "question": q} for i, q in zip(qids, r["questions"])]},
                               ensure_ascii=False) + "\n")
            for i, q, a in zip(qids, r["questions"], r["answers"]):
                g.write(json.dumps({"id": i, "source": src, "question": q, "answers": a}, ensure_ascii=False) + "\n")
    print(f"FactConsolidation: {FC_SOURCES} -> inputs/fc.jsonl, gold/fc_gold.jsonl")
    print(f"work directory: {os.path.abspath(WORK)}")


if __name__ == "__main__":
    main()
