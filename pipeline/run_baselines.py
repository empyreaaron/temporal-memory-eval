"""Baselines without a memory layer: F (full chat history), R (BM25 top-5 sessions), O (evidence sessions only).
Standard library only. Run prepare_data.py first.

Usage (from the repository root):
  python pipeline/run_baselines.py --dry-run          # token/cost estimate only, no key needed
  python pipeline/run_baselines.py --cond O --limit 1 # smoke test: one cheap call
  python pipeline/run_baselines.py                    # O, R, F on all prepared questions (resumable)

Settings come from pipeline/config.example.json, overridden by pipeline/config.local.json (put your key there,
or in the environment variable named by "api_key_env"). Never prints the key; never reads gold/.
Writes <work>/outputs/<cond>.jsonl. Work directory: ./work, or set TME_WORK.
Re-running resumes: the last record per question decides its status (as in the graders), so finished questions
are skipped and failed ones retried. Every record carries a fingerprint of the model settings and the prompt
file; answers made with other settings stop the run instead of being mixed in. A key, balance or request error
(HTTP 400/401/402/403/404/422) stops the run at once. The exit code is non-zero unless every planned call
succeeded. latency_s is the time of the successful API attempt; "attempts" counts tries (the published
baseline runs timed the whole call, retries included).
"""
import argparse, hashlib, json, os, platform, sys, time, urllib.request, urllib.error

PIPE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("TME_WORK", os.path.join(os.path.dirname(PIPE), "work"))
FATAL_STATUS = {400, 401, 402, 403, 404, 422}          # bad request / key / balance: retrying will not help


class FatalAPIError(RuntimeError):
    pass


def load_config():
    with open(os.path.join(PIPE, "config.example.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    local = os.path.join(PIPE, "config.local.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            cfg.update(json.load(f))
    if "extra_params" not in cfg:                     # same answer settings as the mem0 runs
        cfg["extra_params"] = ({"thinking": {"type": "enabled"}, "reasoning_effort": cfg["reasoning_effort"]}
                               if cfg.get("answer_thinking") else {"thinking": {"type": "disabled"}, "temperature": 0})
    key = os.environ.get(cfg.get("api_key_env", ""), "") or cfg.get("api_key", "")
    return cfg, key, "config.local.json" if os.path.exists(local) else "config.example.json"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def latest_by_id(path):                               # last record per id wins (re-runs append)
    return {r["id"]: r for r in load_jsonl(path)}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(cfg, input_sha):
    spec = {"model": cfg["model"], "base_url": cfg["base_url"], "extra_params": cfg["extra_params"], "inputs": input_sha}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def cost(cfg, tin, tout, cache_hit=0):
    """DeepSeek reports cache-hit input tokens separately; they are billed at a much lower rate."""
    hit_price = cfg.get("price_in_cache_hit_per_m", cfg["price_in_per_m"])
    return ((tin - cache_hit) / 1e6 * cfg["price_in_per_m"] + cache_hit / 1e6 * hit_price
            + tout / 1e6 * cfg["price_out_per_m"])


def call(cfg, key, prompt):
    """Returns (response, seconds taken by the successful attempt, number of attempts)."""
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    body.update(cfg["extra_params"])
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    delay = 5
    for attempt in range(1, 7):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 600)) as r:
                return json.loads(r.read().decode("utf-8")), time.time() - t0, attempt
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code in FATAL_STATUS:
                raise FatalAPIError(f"HTTP {e.code}: {msg}")
            if e.code in (429, 500, 502, 503, 504) and attempt < 6:
                print(f"    HTTP {e.code}, retry in {delay}s")
                time.sleep(delay); delay *= 2; continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 6:
                print(f"    network error ({e}), retry in {delay}s")
                time.sleep(delay); delay *= 2; continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cond", default="O,R,F", help="comma-separated conditions (inputs/<cond>.jsonl)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg, key, cfg_name = load_config()
    conds = args.cond.split(",")
    missing = [c for c in conds if not os.path.exists(os.path.join(WORK, "inputs", f"{c}.jsonl"))]
    if missing:
        sys.exit(f"No input file for condition(s) {missing}; run pipeline/prepare_data.py first")
    out_dir = os.path.join(WORK, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    plan, est_in, fps, shas, clash = {}, 0, {}, {}, {}
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for c in conds:
        in_path = os.path.join(WORK, "inputs", f"{c}.jsonl")
        shas[c] = file_sha256(in_path)
        fps[c] = fingerprint(cfg, shas[c])
        rows = load_jsonl(in_path)
        last = latest_by_id(os.path.join(out_dir, f"{c}.jsonl"))
        done = {i for i, o in last.items() if not o.get("error")}
        foreign = sorted(i for i, o in last.items() if not o.get("error") and o.get("fp") != fps[c])
        if foreign:
            clash[c] = foreign
        todo = [r for r in rows if r["id"] not in done][: args.limit]
        plan[c] = todo
        est_in += sum(len(r["prompt"]) / 4 for r in todo)
    if clash:
        sys.exit("These output files hold answers made with other settings or prompts (or by an older version of "
                 "this script), so resuming would mix them:\n"
                 + "\n".join(f"  outputs/{c}.jsonl: {len(ids)} answers, e.g. {ids[:3]}" for c, ids in clash.items())
                 + "\nUse a new work directory (set TME_WORK) or move those files away.")
    est_out = sum(len(v) for v in plan.values()) * cfg.get("est_baseline_out_tokens", 600)
    est = cost(cfg, est_in, est_out)
    print(f"config: {cfg_name} | model: {cfg['model']} | base_url: {cfg['base_url']}")
    for c in conds:
        print(f"  {c}: {len(plan[c])} calls to run")
    print(f"estimate: ~{est_in/1e6:.2f}M input tokens, ~{est_out/1e6:.2f}M output tokens, ~${est:.2f} "
          f"at the configured prices (budget cap ${cfg['budget_usd']:.2f})")
    n_plan = sum(len(v) for v in plan.values())
    if args.dry_run or not n_plan:
        return
    if not key or key.startswith("PASTE"):
        sys.exit("No API key: put it in pipeline/config.local.json or set the environment variable.")
    if est > cfg["budget_usd"]:
        sys.exit("Estimated cost exceeds budget_usd; raise it in pipeline/config.local.json if intended.")
    manifest = os.path.join(out_dir, "run_manifest.jsonl")
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_id, "event": "start", "script": "run_baselines.py", "argv": sys.argv[1:],
                            "config_file": cfg_name, "settings": {k: v for k, v in cfg.items() if k != "api_key"},
                            "fingerprints": {f"{c}.jsonl": fps[c] for c in conds},
                            "inputs_sha256": {f"{c}.jsonl": shas[c] for c in conds},
                            "python": sys.version.split()[0], "platform": platform.platform(),
                            "planned_answers": n_plan}) + "\n")

    spent, n_ok, n_fail, stop = 0.0, 0, 0, None
    for c in conds:
        out_path = os.path.join(out_dir, f"{c}.jsonl")
        for i, r in enumerate(plan[c], 1):
            if stop:
                break
            rec = {"id": r["id"], "cond": c, "model": cfg["model"], "fp": fps[c], "run": run_id,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            t0 = time.time()
            try:
                resp, latency, attempts = call(cfg, key, r["prompt"])
                u = resp.get("usage", {}) or {}
                rec.update(answer=resp["choices"][0]["message"].get("content", ""), api_model=resp.get("model"),
                           prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           cache_hit_tokens=u.get("prompt_cache_hit_tokens"),
                           latency_s=round(latency, 1), attempts=attempts)
                spent += cost(cfg, u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0,
                              u.get("prompt_cache_hit_tokens") or 0)
            except Exception as e:
                rec["error"] = str(e)[:500]
                if isinstance(e, FatalAPIError):
                    stop = f"stopped: {rec['error'][:200]} (key, balance or request problem; retrying will not help)"
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok, n_fail = n_ok + (not rec.get("error")), n_fail + bool(rec.get("error"))
            status = ("ERROR " + rec["error"][:120] if rec.get("error")
                      else f"ok {rec['prompt_tokens']} in / {rec['completion_tokens']} out")
            print(f"[{c} {i}/{len(plan[c])}] {r['id']}: {status} | {time.time()-t0:.1f}s | spent ~${spent:.3f}")
            if spent > cfg["budget_usd"]:
                stop = "budget cap reached"
        if stop:
            break
    n_skip = n_plan - n_ok - n_fail
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_id, "event": "end", "ok": n_ok, "failed": n_fail, "not_run": n_skip,
                            "spent_usd": round(spent, 4)}) + "\n")
    complete = n_ok == n_plan
    print(f"{'done' if complete else 'INCOMPLETE'}: planned {n_plan}, {n_ok} ok, {n_fail} failed, {n_skip} not run | "
          f"actual spend ~${spent:.3f} (from reported usage)")
    if stop:
        print(stop[0].upper() + stop[1:])
    if not complete:
        print("Re-run the same command to retry. Failed calls are not wrong answers: grade only after a complete run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
