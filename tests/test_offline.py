"""Offline checks of the pipeline's bookkeeping: no API key, no network, no mem0 or embedding model needed.

    python -m unittest discover tests
"""
import contextlib, csv, importlib.util, io, json, os, subprocess, sys, tempfile, unittest, urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


runner = module("tme_run_mem0", "pipeline/run_mem0.py")
baselines = module("tme_run_baselines", "pipeline/run_baselines.py")
grade = module("tme_grade", "pipeline/grade.py")
CFG = grade.read_json(os.path.join(ROOT, "pipeline", "config.example.json"))
IDS = grade.read_json(os.path.join(ROOT, "data", "question_ids.json"))
KU, REG = IDS["longmemeval_s_knowledge_update_76"], IDS["longmemeval_s_regression_20"]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class FakeMemory:
    """Two memories from the same day: the newer one (15:00) is more relevant, so mem0 ranks it first."""
    hits = [{"memory": "new", "created_at": "2023-01-01T15:00:00", "score": 0.9},
            {"memory": "undated", "created_at": None, "score": 0.85},
            {"memory": "old", "created_at": "2023-01-01T09:00:00", "score": 0.8}]

    def get_all(self, **kw):
        return {"results": []}

    def search(self, *a, **kw):
        return {"results": [dict(h) for h in self.hits]}


class TestOrdering(unittest.TestCase):
    def run_variants(self, variants):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(CFG, _fp={}, _run="test")
            item = {"id": "q", "type": "knowledge-update", "question": "Q", "question_date": "2023/01/02", "sessions": []}
            prompts = []
            with mock.patch.object(runner, "make_memory", return_value=FakeMemory()):
                runner.lme_item(cfg, "", None, lambda p, rec: prompts.append(p) or "a", item, d, d, d, "patched", variants)
            return {v: read_jsonl(os.path.join(d, f"{v}.jsonl"))[-1] for v in variants}, prompts

    def test_published_order_is_by_day_only(self):
        recs, prompts = self.run_variants(["Mord"])
        self.assertEqual([x["text"] for x in recs["Mord"]["retrieved"]], ["new", "old", "undated"])
        self.assertIn("(Listed from oldest to newest.)", prompts[0])

    def test_full_timestamp_order(self):
        recs, prompts = self.run_variants(["Mord_ts", "M_ts"])
        self.assertEqual([x["text"] for x in recs["Mord_ts"]["retrieved"]], ["old", "new", "undated"])
        self.assertEqual(recs["Mord_ts"]["n_undated"], 1)
        self.assertIn("without a known date come last", prompts[0])
        self.assertIn("- (2023/01/01) old\n- (2023/01/01) new\n- (unknown date) undated", prompts[1])

    def test_offset_times_compare_in_utc(self):
        self.assertLess(runner.time_key("2023-01-01T23:00:00-05:00"), runner.time_key("2023-01-02T05:00:00+00:00"))
        self.assertIsNone(runner.time_key("not a date"))


class TestRunnerBookkeeping(unittest.TestCase):
    def dry_run(self, work, records, fp_override=None):
        item = {"id": "q", "type": "knowledge-update", "sessions": []}
        cfg = dict(CFG, fc_conditions=[], lme_variants=["M"], lme_default_usage=0)
        out = io.StringIO()
        write_jsonl(os.path.join(work, "outputs", "M.jsonl"), records)
        with mock.patch.object(runner, "WORK", work), \
             mock.patch.object(runner, "load_config", return_value=(cfg, "", "fixture")), \
             mock.patch.object(runner, "build_plan", return_value=([], [item], [])), \
             mock.patch.object(sys, "argv", ["run_mem0.py", "--task", "lme", "--dry-run"]), \
             contextlib.redirect_stdout(out):
            runner.main()
        return out.getvalue()

    def current_fp(self):
        cfg = dict(CFG, fc_conditions=[], lme_variants=["M"])
        return runner.output_fingerprints(cfg, {})["M.jsonl"]

    def test_last_record_decides(self):
        fp = self.current_fp()
        with tempfile.TemporaryDirectory() as d:
            out = self.dry_run(d, [{"id": "q", "answer": "x", "fp": fp}, {"id": "q", "error": "HTTP 500", "fp": fp}])
            self.assertIn("answers to produce: 1", out)          # success then failure: retried
            out = self.dry_run(d, [{"id": "q", "error": "HTTP 500", "fp": fp}, {"id": "q", "answer": "x", "fp": fp}])
            self.assertIn("answers to produce: 0", out)          # failure then success: done

    def test_answers_from_other_settings_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                self.dry_run(d, [{"id": "q", "answer": "x", "fp": "something-else"}])
            self.assertIn("other settings", str(e.exception.code))

    def test_store_reuse_checks_settings_and_input(self):
        with tempfile.TemporaryDirectory() as d:
            settings = runner.store_settings(CFG, "patched")
            runner.write_store_config(d, settings, [{"date": "2023/01/01"}])
            runner.check_store(d, settings, [{"date": "2023/01/01"}])
            with self.assertRaises(RuntimeError):
                runner.check_store(d, settings, [{"date": "2023/01/02"}])
            with self.assertRaises(RuntimeError):
                runner.check_store(d, dict(settings, model="other"), [{"date": "2023/01/01"}])

    def test_status_separates_failed_and_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"_run": "now"}
            write_jsonl(os.path.join(d, "M.jsonl"), [{"id": "a", "answer": "x", "run": "now"},
                                                     {"id": "b", "error": "e", "run": "now"},
                                                     {"id": "c", "error": "e", "run": "earlier"}])
            counts = runner.run_status(cfg, d, ["M.jsonl"], {("M.jsonl", i) for i in "abcd"})
            self.assertEqual((counts["ok"], counts["failed"], counts["not run"]), (1, 1, 2))


class TestBaselineFailure(unittest.TestCase):
    def test_key_error_stops_and_fails(self):
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "inputs", "F.jsonl"), [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "p"}])
            cfg, _, _ = baselines.load_config()
            err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"bad key"))
            out = io.StringIO()
            with mock.patch.object(baselines, "WORK", d), \
                 mock.patch.object(baselines, "load_config", return_value=(cfg, "k", "fixture")), \
                 mock.patch.object(baselines.urllib.request, "urlopen", side_effect=err) as calls, \
                 mock.patch.object(sys, "argv", ["run_baselines.py", "--cond", "F"]), \
                 contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as e:
                baselines.main()
            self.assertEqual(e.exception.code, 1)
            self.assertEqual(calls.call_count, 1)                # no retries, no second question
            self.assertIn("INCOMPLETE: planned 2, 0 ok, 1 failed, 1 not run", out.getvalue())
            self.assertFalse(any(l.startswith("done") for l in out.getvalue().splitlines()))


class TestGrading(unittest.TestCase):
    GOLD = [{"id": "a", "question": "Where?", "answer": "the closet"},
            {"id": "b", "question": "How many women?", "answer": "6 women"}]

    def setup_work(self, d, answers):
        write_jsonl(os.path.join(d, "gold", "lme_gold.jsonl"), self.GOLD)
        write_jsonl(os.path.join(d, "gold", "fc_gold.jsonl"), [])
        write_jsonl(os.path.join(d, "outputs", "M.jsonl"), answers)
        return mock.patch.multiple(grade, WORK=d, OUT=os.path.join(d, "outputs"), GRADES=os.path.join(d, "grades"),
                                   expected_ids=lambda: {c: ["a", "b"] for c in grade.LME_CONDS})

    def fill_sheet(self, d, verdicts, drop=(), dup=()):
        path = os.path.join(d, "grades", "review_sheet.csv")
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [dict(r, verdict=verdicts) for r in rows if r["n"] not in drop] + [r for r in rows if r["n"] in dup]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n", "question", "question_date", "gold", "answer", "verdict"])
            w.writeheader(); w.writerows(rows)

    def test_hedges_and_number_only_matches_go_to_review(self):
        answers = [{"id": "a", "answer": "Under the bed; you planned to move them to the closet, but that is not confirmed."},
                   {"id": "b", "answer": "6 men."}]
        with tempfile.TemporaryDirectory() as d, self.setup_work(d, answers), contextlib.redirect_stdout(io.StringIO()):
            grade.auto()
            key = grade.read_json(os.path.join(d, "grades", "review_key.json"))
            self.assertEqual(sorted(k["id"] for k in key), ["a", "b"])

    def test_apply_needs_every_row_once(self):
        answers = [{"id": "a", "answer": "planned closet, not confirmed"}, {"id": "b", "answer": "6 men"}]
        with tempfile.TemporaryDirectory() as d, self.setup_work(d, answers), contextlib.redirect_stdout(io.StringIO()):
            grade.auto()
            self.fill_sheet(d, "0", drop=("2",))
            with self.assertRaisesRegex(SystemExit, "rows missing"):
                grade.apply()
            grade.auto()
            self.fill_sheet(d, "0", dup=("1",))
            with self.assertRaisesRegex(SystemExit, "duplicated"):
                grade.apply()
            grade.auto()
            self.fill_sheet(d, "0")
            grade.apply()
            detail = read_jsonl(os.path.join(d, "grades", "lme_final_detail.jsonl"))
            self.assertEqual({x["source"] for x in detail}, {"review"})
            self.assertEqual(grade.read_json(os.path.join(d, "grades", "lme_final.json")), {"M|a": False, "M|b": False})

    def test_stale_sheet_is_rejected(self):
        answers = [{"id": "a", "answer": "planned closet, not confirmed"}, {"id": "b", "answer": "6 men"}]
        with tempfile.TemporaryDirectory() as d, self.setup_work(d, answers), contextlib.redirect_stdout(io.StringIO()):
            grade.auto()
            self.fill_sheet(d, "1")
            with open(os.path.join(d, "outputs", "M.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": "b", "answer": "6 women"}) + "\n")
            with self.assertRaisesRegex(SystemExit, "changed after grade.py auto"):
                grade.apply()

    def test_api_errors_are_not_graded(self):
        answers = [{"id": "a", "answer": "the closet"}, {"id": "b", "error": "HTTP 401"}]
        with tempfile.TemporaryDirectory() as d, self.setup_work(d, answers), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "Not graded"):
                grade.auto()
            grade.auto(allow_incomplete=True)
            self.assertEqual([r["id"] for r in read_jsonl(os.path.join(d, "grades", "lme_auto.jsonl"))], ["a"])
            self.fill_sheet(d, "1")
            with self.assertRaisesRegex(SystemExit, "incomplete"):
                grade.apply()

    def test_fc_normalization_matches_memoryagentbench(self):
        self.assertEqual(grade.fc_norm("The answer: U.S.A.!"), "answer usa")
        self.assertIn(grade.fc_norm("13"), grade.fc_norm("130"))     # substring rule, disclosed in the README


FC_ALL = ["FULL", "BM25", "MEM0", "MEM0_r2", "MEM0_T", "MEM0_S"]
FC_IDS = [("s6", "factconsolidation_sh_6k"), ("s32", "factconsolidation_sh_32k")]


class TestSummaryRefusesIncompleteResults(unittest.TestCase):
    def run_summary(self, d, fc_rows, calls=()):
        conds = ["M0", "Mnd", "Mord", "M", "M_r2", "M_r3", "F", "R", "O"]
        grades = {f"{c}|{q}": True for c in conds for q in KU} | \
                 {f"{c}|{q}": True for c in ["M0", "Mnd", "Mord", "M", "F", "R", "O"] for q in REG}
        with open(os.path.join(d, "grades.json"), "w") as f:
            json.dump(grades, f)
        write_jsonl(os.path.join(d, "questions.jsonl"),
                    [{"id": q, "type": "knowledge-update"} for q in KU] + [{"id": q, "type": "multi-session"} for q in REG])
        write_jsonl(os.path.join(d, "fc_gold.jsonl"), [{"id": "s6", "source": "factconsolidation_sh_6k"},
                                                       {"id": "s32", "source": "factconsolidation_sh_32k"}])
        write_jsonl(os.path.join(d, "fc_graded.jsonl"), fc_rows)
        write_jsonl(os.path.join(d, "calls.jsonl"), list(calls))
        return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize_results.py"),
                               "--grades", os.path.join(d, "grades.json"), "--questions", os.path.join(d, "questions.jsonl"),
                               "--fc-graded", os.path.join(d, "fc_graded.jsonl"), "--fc-gold", os.path.join(d, "fc_gold.jsonl"),
                               "--c1-outputs", d, "--m1-calls", os.path.join(d, "calls.jsonl"), "--out", os.path.join(d, "out")],
                              capture_output=True, text=True)

    def test_missing_fc_condition_is_an_error_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_summary(d, [{"id": "s6", "source": "factconsolidation_sh_6k", "cond": "FULL", "correct": True}])
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("incomplete", p.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "out", "factconsolidation_summary.csv")))

    def test_question_missing_from_every_condition_is_caught(self):
        rows = [{"id": "s32", "source": "factconsolidation_sh_32k", "cond": c, "correct": True} for c in FC_ALL]
        with tempfile.TemporaryDirectory() as d:
            p = self.run_summary(d, rows)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("1 missing", p.stderr)

    def test_complete_fc_results_are_summarized(self):
        rows = [{"id": i, "source": s, "cond": c, "correct": True} for c in FC_ALL for i, s in FC_IDS]
        with tempfile.TemporaryDirectory() as d:
            p = self.run_summary(d, rows)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(d, "out", "factconsolidation_summary.csv"), encoding="utf-8") as f:
                summary = f.read()
            self.assertIn("total_of_2", summary)


    def test_telemetry_counts_only_the_call_behind_the_graded_answer(self):
        q = KU[0]
        rows = [{"id": i, "source": s, "cond": c, "correct": True} for c in FC_ALL for i, s in FC_IDS]
        call = lambda item, ts, n, tokens, **kw: {"kind": "answer", "item": item, "ts": ts, "out_chars": n,
                                                  "prompt_tokens": tokens, "reasoning_tokens": 0, "latency_s": 1.0, **kw}
        with tempfile.TemporaryDirectory() as d:
            # current runner: success, failure, success; the call ids say which call produced the graded answer
            write_jsonl(os.path.join(d, "M.jsonl"), [
                {"id": q, "answer": "a", "call_id": "c1", "ts": "2026-01-01 10:00:00"},
                {"id": q, "error": "HTTP 500", "call_id": "c2", "ts": "2026-01-01 10:05:00"},
                {"id": q, "answer": "bb", "call_id": "c3", "ts": "2026-01-01 10:10:00"}])
            # first-release runner: variant tag but no call id; matched by time and answer length
            write_jsonl(os.path.join(d, "Mnd.jsonl"), [
                {"id": q, "answer": "xx", "ts": "2026-01-01 10:00:00"},
                {"id": q, "answer": "yyy", "ts": "2026-01-01 11:00:00"}])
            p = self.run_summary(d, rows, [
                call(f"lme:{q}:M", "2026-01-01 10:00:03", 1, 100, call_id="c1"),
                call(f"lme:{q}:M", "2026-01-01 10:10:03", 2, 900, call_id="c3"),
                call(f"lme:{q}:Mnd", "2026-01-01 10:00:03", 2, 200),
                call(f"lme:{q}:Mnd", "2026-01-01 11:00:03", 3, 300)])
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(os.path.join(d, "out", "longmemeval_ku76_summary.csv"), encoding="utf-8", newline="") as f:
                tele = {r["condition"]: r for r in csv.DictReader(f)}
            self.assertEqual((tele["M"]["mean_answer_input_tokens"], tele["M"]["telemetry_n"]), ("900", "1"))
            self.assertEqual((tele["Mnd"]["mean_answer_input_tokens"], tele["Mnd"]["telemetry_n"]), ("300", "1"))


class TestExportChecksBothWays(unittest.TestCase):
    def run_export(self, d, fc_answers, fc_graded, lme_grades=None, lme_answers=None):
        write_jsonl(os.path.join(d, "fc", "fc_FULL.jsonl"), fc_answers)
        write_jsonl(os.path.join(d, "fc_graded.jsonl"), fc_graded)
        write_jsonl(os.path.join(d, "lme", "M.jsonl"), lme_answers or [])
        with open(os.path.join(d, "grades.json"), "w") as f:
            json.dump(lme_grades or {}, f)
        return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_answers.py"),
                               "--grades", os.path.join(d, "grades.json"), "--fc-graded", os.path.join(d, "fc_graded.jsonl"),
                               "--lme-outputs", os.path.join(d, "lme"), "--fc-outputs", os.path.join(d, "fc"),
                               "--out", os.path.join(d, "out")], capture_output=True, text=True)

    def check_refused(self, d, p, text):
        self.assertNotEqual(p.returncode, 0)
        self.assertIn(text, p.stderr)
        self.assertFalse(os.path.exists(os.path.join(d, "out")))      # nothing half-written

    def test_grade_without_answer(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_export(d, [], [{"id": "x", "source": "s", "cond": "FULL", "correct": True}])
            self.check_refused(d, p, "grade has no answer")

    def test_failed_last_record_keeps_no_old_grade(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_export(d, [{"id": "x", "source": "s", "answer": "ok"}, {"id": "x", "source": "s", "error": "HTTP 401"}],
                                [{"id": "x", "source": "s", "cond": "FULL", "correct": True}])
            self.check_refused(d, p, "API error but it is graded")

    def test_duplicate_grade(self):
        with tempfile.TemporaryDirectory() as d:
            g = {"id": "x", "source": "s", "cond": "FULL", "correct": True}
            p = self.run_export(d, [{"id": "x", "source": "s", "answer": "ok"}], [g, g])
            self.check_refused(d, p, "graded 2 times")

    def test_longmemeval_failed_answer(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_export(d, [], [], {f"M|{KU[0]}": True}, [{"id": KU[0], "error": "HTTP 500"}])
            self.check_refused(d, p, "API error but it is graded")


if __name__ == "__main__":
    unittest.main()
