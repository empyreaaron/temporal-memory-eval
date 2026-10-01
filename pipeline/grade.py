"""Grade the runs in two steps. Run from the repository root (work directory: ./work, or set TME_WORK).

  python pipeline/grade.py auto
      1. FactConsolidation: substring match against the gold answers (as in MemoryAgentBench).
      2. LongMemEval: automatic first pass, then a blind review sheet of every answer the rules cannot settle,
         plus answers graded correct that look suspicious (extra numbers, or long enough to hedge).
      Writes <work>/grades/{fc_graded.jsonl, lme_auto.jsonl, review_sheet.csv, review_key.json}.

  (a person fills the "verdict" column of review_sheet.csv with 1 = correct or 0 = wrong;
   condition labels are hidden and rows are shuffled)

  python pipeline/grade.py apply
      Merges the verdicts into <work>/grades/lme_final.json ("<condition>|<question_id>" -> true/false),
      which scripts/summarize_results.py turns into results/*.csv.

Review rules used for the published results:
  - the latest value with a caveat that it may not be current: correct
  - a correct answer that also mentions the old value: correct
  - "the records conflict" without choosing: wrong
  - abstention questions: correct only if the answer says the premise was never mentioned
"""
import csv, json, os, random, re, string, sys

PIPE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("TME_WORK", os.path.join(os.path.dirname(PIPE), "work"))
OUT, GRADES = os.path.join(WORK, "outputs"), os.path.join(WORK, "grades")
LME_CONDS = ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "F", "R", "O"]
FC_CONDS = ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]
MISSING = ["not enough information", "no information", "not specified", "does not contain", "doesn't contain",
           "does not mention", "doesn't mention", "no mention", "not mentioned", "cannot determine",
           "can't determine", "cannot be determined", "unable to determine", "not stated", "never stated",
           "don't have enough", "do not have enough", "not provided", "no record", "don't mention",
           "do not mention", "didn't mention", "did not mention", "nothing about", "no reference to"]
WORD_NUM = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
STOP = {"in", "my", "your", "their", "his", "her", "its", "our", "of", "on", "at", "to", "is", "was", "and", "i"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def latest(path):                                     # last record per id wins (re-runs append)
    return {r["id"]: r for r in load_jsonl(path)}


def norm(s):
    s = str(s if s is not None else "").lower().replace("**", "").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'")
    for w, d in WORD_NUM.items():
        s = re.sub(rf"\b{w}\b", d, s)
    s = re.sub(r"\.(?!\d)", " ", s)                   # keep decimals, drop sentence periods
    s = "".join(ch if ch not in set(string.punctuation) - {"-", "."} else " " for ch in s)
    s = re.sub(r"\b(a|an|the|about|approximately)\b", " ", s)
    return " ".join(s.split())


def nums(s):
    return re.findall(r"\d+(?:\.\d+)?", norm(s))


def judge(qid, gold, ans):
    """Returns (correct, needs_review, reason)."""
    a, g = norm(ans), norm(gold)
    says_missing = any(p in a for p in MISSING)
    if qid.endswith("_abs"):
        return says_missing, not says_missing, "" if says_missing else "abstention: no missing-info phrase"
    if g.split()[:1] in (["yes"], ["no"]):
        want, other = g.split()[0], ("no" if g.startswith("yes") else "yes")
        head = a.split()[:12]
        if want in head and other not in head:
            return True, False, ""
        return False, True, "yes/no question: answer does not open with a clear verdict"
    gn, words = nums(gold), [w for w in g.split() if w not in STOP]
    hit = (g and g in a) or (gn and all(n in nums(ans) for n in gn)) \
          or (0 < len(words) <= 4 and all(w in a.split() for w in words))
    if hit and says_missing:
        return False, True, "hedged: has the value but also says information is missing"
    if hit:
        return True, False, ""
    return False, not says_missing, "" if says_missing else "no match; check for a paraphrase"


def numset(s):
    s = str(s).lower()
    for w, d in WORD_NUM.items():
        s = re.sub(rf"\b{w}\b", d, s)
    return {n for n in re.findall(r"\d+(?:\.\d+)?", s) if not re.fullmatch(r"(19|20)\d\d", n)}


def auto():
    os.makedirs(GRADES, exist_ok=True)
    fc_gold = {g["id"]: g for g in load_jsonl(os.path.join(WORK, "gold", "fc_gold.jsonl"))}
    with open(os.path.join(GRADES, "fc_graded.jsonl"), "w", encoding="utf-8") as f:
        for c in FC_CONDS:
            for qid, r in latest(os.path.join(OUT, f"fc_{c}.jsonl")).items():
                g = fc_gold[qid]
                ok = not r.get("error") and any(norm(a) in norm(r.get("answer", "")) for a in g["answers"])
                f.write(json.dumps({"id": qid, "source": g["source"], "cond": c, "correct": ok}) + "\n")
    gold = {g["id"]: g for g in load_jsonl(os.path.join(WORK, "gold", "lme_gold.jsonl"))}
    rows, review = [], []
    for c in LME_CONDS:
        for qid, r in latest(os.path.join(OUT, f"{c}.jsonl")).items():
            g = gold[qid]["answer"]
            correct, needs, reason = ((False, True, "error: " + r["error"][:80]) if r.get("error")
                                      else judge(qid, g, r.get("answer", "")))
            extra = numset(r.get("answer", "")) - numset(g) - numset(gold[qid]["question"])
            suspicious = correct and (bool(extra) or len(r.get("answer", "")) > 220)
            row = {"id": qid, "cond": c, "auto_correct": correct, "reason": reason}
            rows.append(row)
            if needs or suspicious:
                review.append((row, gold[qid], r.get("answer", "")))
    with open(os.path.join(GRADES, "lme_auto.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    random.Random(20261001).shuffle(review)
    with open(os.path.join(GRADES, "review_sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "question", "question_date", "gold", "answer", "verdict"])
        for n, (row, g, ans) in enumerate(review, 1):
            w.writerow([n, g["question"], g.get("question_date", ""), g["answer"], ans.strip(), ""])
    json.dump([{"n": n, "id": row["id"], "cond": row["cond"], "auto_correct": row["auto_correct"]}
               for n, (row, _, _) in enumerate(review, 1)],
              open(os.path.join(GRADES, "review_key.json"), "w"), indent=0)
    print(f"FactConsolidation graded -> grades/fc_graded.jsonl")
    print(f"LongMemEval: {len(rows)} answers, {len(review)} to review by hand -> grades/review_sheet.csv")


def apply():
    key = {k["n"]: k for k in json.load(open(os.path.join(GRADES, "review_key.json")))}
    final = {f"{r['cond']}|{r['id']}": r["auto_correct"] for r in load_jsonl(os.path.join(GRADES, "lme_auto.jsonl"))}
    blank = []
    with open(os.path.join(GRADES, "review_sheet.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row["verdict"].strip()
            if v not in ("0", "1"):
                blank.append(row["n"]); continue
            k = key[int(row["n"])]
            final[f"{k['cond']}|{k['id']}"] = v == "1"
    if blank:
        sys.exit(f"{len(blank)} rows of review_sheet.csv have no verdict (expected 1 or 0), e.g. n={blank[:5]}")
    json.dump(final, open(os.path.join(GRADES, "lme_final.json"), "w"), indent=0)
    print(f"wrote grades/lme_final.json ({len(final)} graded answers)")


if __name__ == "__main__":
    {"auto": auto, "apply": apply}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
