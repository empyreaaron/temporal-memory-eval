"""Grade the runs in two steps. Run from the repository root (work directory: ./work, or set TME_WORK).

  python pipeline/grade.py auto [--allow-incomplete]
      1. FactConsolidation: substring match against the gold answers, with MemoryAgentBench's SQuAD-style
         normalization (so a gold "13" also matches inside "130"; none of the published verdicts depends on
         such a partial-token match).
      2. LongMemEval: automatic first pass, then a blind review sheet of every answer the rules cannot settle,
         plus answers graded correct that look suspicious: extra numbers, long enough to hedge, hedging or
         conflict wording ("does not confirm", "planned", "however", ...), a number match without the gold's
         words ("6 men" for "6 women"), or an abstention that goes on to state facts.
      First checks that every condition that was run is complete: each expected question has a final record
      (the last record per id, as in the runners) and it is not an API error. Failed or missing answers are not
      graded as wrong; the command stops instead. --allow-incomplete grades what is there (for inspection
      only: the grades are marked incomplete and the summary refuses to use them).
      Writes <work>/grades/{fc_graded.jsonl, lme_auto.jsonl, review_sheet.csv, review_key.json, coverage.json}.

  (a person fills the "verdict" column of review_sheet.csv with 1 = correct or 0 = wrong;
   condition labels are hidden and rows are shuffled)

  python pipeline/grade.py apply
      Checks that the sheet has exactly one filled verdict for every row of review_key.json and still matches
      the current answers, then writes <work>/grades/lme_final.json ("<condition>|<question_id>" -> true/false)
      and lme_final_detail.jsonl (each grade with its source: automatic or review).
      scripts/summarize_results.py turns lme_final.json into results/*.csv.

Review rules used for the published results:
  - the latest value with a caveat that it may not be current: correct
  - a correct answer that also mentions the old value: correct
  - listing the old and the new value without choosing, or "the records conflict": wrong
  - an old value and a later plan to change it, reported without saying which holds now: wrong
  - abstention questions: correct only if the answer says the premise was never mentioned
"""
import collections, csv, hashlib, json, os, random, re, string, sys

PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PIPE)
WORK = os.environ.get("TME_WORK", os.path.join(ROOT, "work"))
OUT, GRADES = os.path.join(WORK, "outputs"), os.path.join(WORK, "grades")
LME_CONDS = ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "Mord_ts", "M_ts", "F", "R", "O"]
REGRESSION_CONDS = {"M0", "Mnd", "Mord", "M", "F", "R", "O"}     # also expected on the 20 regression questions
FC_CONDS = ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]
MISSING = ["not enough information", "no information", "not specified", "does not contain", "doesn't contain",
           "does not mention", "doesn't mention", "no mention", "not mentioned", "cannot determine",
           "can't determine", "cannot be determined", "unable to determine", "not stated", "never stated",
           "don't have enough", "do not have enough", "not provided", "no record", "don't mention",
           "do not mention", "didn't mention", "did not mention", "nothing about", "no reference to"]
# wording that often marks an answer that hedges between an old and a new value; sends it to review only
HEDGE = ["does not confirm", "doesn't confirm", "do not confirm", "don't confirm", "not confirmed", "unconfirmed",
         "unclear", "not sure", "uncertain", "conflict", "contradict", "however", "although", "no longer",
         "used to", "previously", "originally", "at first", "first said", "planned", "planning", "plan to",
         "rather than", "instead of", "either"]
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


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def answer_hash(r):
    return hashlib.sha256((r.get("answer") or "").encode("utf-8")).hexdigest()[:16]


def norm(s):
    s = str(s if s is not None else "").lower().replace("**", "").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'")
    for w, d in WORD_NUM.items():
        s = re.sub(rf"\b{w}\b", d, s)
    s = re.sub(r"\.(?!\d)", " ", s)                   # keep decimals, drop sentence periods
    s = "".join(ch if ch not in set(string.punctuation) - {"-", "."} else " " for ch in s)
    s = re.sub(r"\b(a|an|the|about|approximately)\b", " ", s)
    return " ".join(s.split())


def fc_norm(s):
    """MemoryAgentBench / SQuAD normalization, exactly as used for the published FactConsolidation grades."""
    s = str(s if s is not None else "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
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


def suspicion(qid, gold, ans):
    """Why an answer graded correct still goes to review ('' if it does not)."""
    a, g = norm(ans), norm(gold["answer"])
    if numset(ans) - numset(gold["answer"]) - numset(gold["question"]):
        return "extra numbers"
    if len(ans) > 220:
        return "long answer"
    if any(h in a for h in HEDGE):
        return "hedging or conflict wording"
    if qid.endswith("_abs"):
        if numset(ans) - numset(gold["question"]) or any(w in a.split() for w in ("but", "however", "although")):
            return "abstention that goes on to state facts"
        return ""
    words = [w for w in g.split() if w not in STOP and not re.fullmatch(r"[\d.]+", w)]
    if nums(gold["answer"]) and g not in a and words and not all(w in a.split() for w in words):
        return "number matches but the gold's words do not"
    return ""


def expected_ids():
    ids = read_json(os.path.join(ROOT, "data", "question_ids.json"))
    ku, reg = ids["longmemeval_s_knowledge_update_76"], ids["longmemeval_s_regression_20"]
    return {c: ku + (reg if c in REGRESSION_CONDS else []) for c in LME_CONDS}


def check_file(path, expect):
    """Coverage of one condition: final records for the expected ids, API errors, mixed fingerprints."""
    if not os.path.exists(path):
        return None, {}
    last = latest(path)
    errors = sorted(i for i, r in last.items() if r.get("error"))
    missing = sorted(set(expect) - set(last))
    fps = {r.get("fp") for r in last.values() if not r.get("error")}
    info = {"expected": len(expect), "answered": len(set(expect) & set(last)) - len(set(expect) & set(errors)),
            "errors": errors, "missing": missing, "mixed_settings": len(fps) > 1}
    return info, last


def auto(allow_incomplete=False):
    os.makedirs(GRADES, exist_ok=True)
    fc_gold = {g["id"]: g for g in load_jsonl(os.path.join(WORK, "gold", "fc_gold.jsonl"))}
    gold = {g["id"]: g for g in load_jsonl(os.path.join(WORK, "gold", "lme_gold.jsonl"))}
    exp = expected_ids()
    coverage, fc_runs, lme_runs = {"fc": {}, "lme": {}}, {}, {}
    for c in FC_CONDS:
        info, last = check_file(os.path.join(OUT, f"fc_{c}.jsonl"), list(fc_gold))
        if info:
            coverage["fc"][c], fc_runs[c] = info, last
    for c in LME_CONDS:
        info, last = check_file(os.path.join(OUT, f"{c}.jsonl"), exp[c])
        if info:
            coverage["lme"][c], lme_runs[c] = info, last
    bad = {f"{t}:{c}": i for t in ("fc", "lme") for c, i in coverage[t].items()
           if i["errors"] or i["missing"] or i["mixed_settings"]}
    coverage["complete"] = not bad
    coverage["not_run"] = {"fc": [c for c in FC_CONDS if c not in fc_runs], "lme": [c for c in LME_CONDS if c not in lme_runs]}
    for name, i in bad.items():
        print(f"INCOMPLETE {name}: {i['answered']}/{i['expected']} answered, {len(i['errors'])} API errors "
              f"(e.g. {i['errors'][:3]}), {len(i['missing'])} missing (e.g. {i['missing'][:3]})"
              + (", answers made with different settings" if i["mixed_settings"] else ""))
    if bad and not allow_incomplete:
        sys.exit("Not graded: re-run the runner until it reports done (failed calls are not wrong answers), "
                 "or pass --allow-incomplete to inspect partial results.")
    with open(os.path.join(GRADES, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=1)

    with open(os.path.join(GRADES, "fc_graded.jsonl"), "w", encoding="utf-8") as f:
        for c, last in fc_runs.items():
            for qid, r in last.items():
                if r.get("error"):
                    continue                          # never graded: an API failure is not a wrong answer
                g = fc_gold[qid]
                ok = any(fc_norm(a) in fc_norm(r.get("answer", "")) for a in g["answers"])
                f.write(json.dumps({"id": qid, "source": g["source"], "cond": c, "correct": ok}) + "\n")
    rows, review = [], []
    for c, last in lme_runs.items():
        for qid, r in last.items():
            if r.get("error"):
                continue
            g, ans = gold[qid], r.get("answer", "")
            correct, needs, reason = judge(qid, g["answer"], ans)
            why = suspicion(qid, g, ans) if correct else ""
            row = {"id": qid, "cond": c, "auto_correct": correct, "reason": reason or why, "answer_sha": answer_hash(r)}
            rows.append(row)
            if needs or why:
                review.append((row, g, ans))
    with open(os.path.join(GRADES, "lme_auto.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    random.Random(20261001).shuffle(review)
    with open(os.path.join(GRADES, "review_sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "question", "question_date", "gold", "answer", "verdict"])
        for n, (row, g, ans) in enumerate(review, 1):
            w.writerow([n, g["question"], g.get("question_date", ""), g["answer"], ans.strip(), ""])
    with open(os.path.join(GRADES, "review_key.json"), "w", encoding="utf-8") as f:
        json.dump([{"n": n, "id": row["id"], "cond": row["cond"], "auto_correct": row["auto_correct"]}
                   for n, (row, _, _) in enumerate(review, 1)], f, indent=0)
    print(f"FactConsolidation graded -> grades/fc_graded.jsonl")
    print(f"LongMemEval: {len(rows)} answers, {len(review)} to review by hand -> grades/review_sheet.csv")
    if bad:
        print("Grades are marked incomplete (grades/coverage.json); they are for inspection only.")


def same_text(a, b):
    """Loose comparison that survives spreadsheet re-saving (whitespace, non-ASCII characters)."""
    clean = lambda s: " ".join("".join(ch for ch in str(s) if ord(ch) < 128).split())
    return clean(a) == clean(b)


def apply():
    key_rows = read_json(os.path.join(GRADES, "review_key.json"))
    key = {k["n"]: k for k in key_rows}
    if len(key) != len(key_rows):
        sys.exit("review_key.json has duplicate row numbers; re-run grade.py auto")
    coverage = read_json(os.path.join(GRADES, "coverage.json"))
    auto_rows = load_jsonl(os.path.join(GRADES, "lme_auto.jsonl"))
    gold = {g["id"]: g for g in load_jsonl(os.path.join(WORK, "gold", "lme_gold.jsonl"))}
    answers = {c: latest(os.path.join(OUT, f"{c}.jsonl")) for c in LME_CONDS}
    now = {(c, i, answer_hash(r)) for c, last in answers.items() for i, r in last.items() if not r.get("error")}
    if {(r["cond"], r["id"], r.get("answer_sha")) for r in auto_rows} != now:
        sys.exit("The answer files changed after grade.py auto; run auto again and review the new sheet.")

    with open(os.path.join(GRADES, "review_sheet.csv"), encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    numbers = [int(r["n"]) for r in sheet if (r.get("n") or "").strip().isdigit()]
    dup = sorted(n for n, k in collections.Counter(numbers).items() if k > 1)
    absent, extra = sorted(set(key) - set(numbers)), sorted(set(numbers) - set(key))
    if len(numbers) != len(sheet) or dup or absent or extra:
        sys.exit(f"review_sheet.csv does not match review_key.json: rows missing {absent[:10]}, duplicated "
                 f"{dup[:10]}, unknown {extra[:10]}, {len(sheet) - len(numbers)} rows without a number. "
                 "Every row of the sheet needs exactly one verdict.")
    blank, stale, verdict = [], [], {}
    for row in sheet:
        n = int(row["n"])
        k = key[n]
        current = answers.get(k["cond"], {}).get(k["id"], {})
        if not (same_text(row["question"], gold.get(k["id"], {}).get("question"))
                and same_text(row["answer"], current.get("answer", "").strip())):
            stale.append(n)
        v = (row.get("verdict") or "").strip()
        if v not in ("0", "1"):
            blank.append(n)
        verdict[n] = v == "1"
    if stale:
        sys.exit(f"{len(stale)} rows of review_sheet.csv do not match the current answers (e.g. n={stale[:5]}); "
                 "the sheet is from an older run. Run grade.py auto again.")
    if blank:
        sys.exit(f"{len(blank)} rows of review_sheet.csv have no verdict (expected 1 or 0), e.g. n={blank[:5]}")
    if not coverage.get("complete"):
        sys.exit("grades/coverage.json says the runs were incomplete; final grades are not written. "
                 "Finish the runs, then run grade.py auto and apply again.")

    reviewed = {(key[n]["cond"], key[n]["id"]): v for n, v in verdict.items()}
    final, detail = {}, []
    for r in auto_rows:
        k = (r["cond"], r["id"])
        final[f"{r['cond']}|{r['id']}"] = reviewed.get(k, r["auto_correct"])
        detail.append({"cond": r["cond"], "id": r["id"], "final": final[f"{r['cond']}|{r['id']}"],
                       "source": "review" if k in reviewed else "automatic",
                       "auto_correct": r["auto_correct"], "reason": r.get("reason", "")})
    with open(os.path.join(GRADES, "lme_final.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=0)
    with open(os.path.join(GRADES, "lme_final_detail.jsonl"), "w", encoding="utf-8") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")
    print(f"wrote grades/lme_final.json ({len(final)} graded answers, {len(reviewed)} from the review sheet)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "auto":
        auto("--allow-incomplete" in sys.argv[2:])
    elif cmd == "apply":
        apply()
    else:
        sys.exit(__doc__)
