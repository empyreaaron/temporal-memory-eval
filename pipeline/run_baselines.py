"""Baselines without a memory layer: F (full chat history), R (BM25 top-5 sessions), O (evidence sessions only).
Standard library only. Run prepare_data.py first.

Usage (from the repository root):
  python pipeline/run_baselines.py --dry-run          # token/cost estimate only, no key needed
  python pipeline/run_baselines.py --cond O --limit 1 # smoke test: one cheap call
  python pipeline/run_baselines.py                    # O, R, F on all prepared questions (resumable)

Settings come from pipeline/config.example.json, overridden by pipeline/config.local.json (put your key there,
or in the environment variable named by "api_key_env"). Never prints the key; never reads gold/.
Writes <work>/outputs/<cond>.jsonl. Work directory: ./work, or set TME_WORK.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

PIPE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("TME_WORK", os.path.join(os.path.dirname(PIPE), "work"))


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


def cost(cfg, tin, tout, cache_hit=0):
    """DeepSeek reports cache-hit input tokens separately; they are billed at a much lower rate."""
    hit_price = cfg.get("price_in_cache_hit_per_m", cfg["price_in_per_m"])
    return ((tin - cache_hit) / 1e6 * cfg["price_in_per_m"] + cache_hit / 1e6 * hit_price
            + tout / 1e6 * cfg["price_out_per_m"])


def call(cfg, key, prompt):
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    body.update(cfg["extra_params"])
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    delay = 5
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 600)) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                print(f"    HTTP {e.code}, retry in {delay}s")
                time.sleep(delay); delay *= 2; continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
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
    os.makedirs(os.path.join(WORK, "outputs"), exist_ok=True)

    plan, est_in = {}, 0
    for c in conds:
        rows = load_jsonl(os.path.join(WORK, "inputs", f"{c}.jsonl"))
        done = {o["id"] for o in load_jsonl(os.path.join(WORK, "outputs", f"{c}.jsonl")) if not o.get("error")}
        todo = [r for r in rows if r["id"] not in done][: args.limit]
        plan[c] = todo
        est_in += sum(len(r["prompt"]) / 4 for r in todo)
    est_out = sum(len(v) for v in plan.values()) * cfg.get("est_baseline_out_tokens", 600)
    est = cost(cfg, est_in, est_out)
    print(f"config: {cfg_name} | model: {cfg['model']} | base_url: {cfg['base_url']}")
    for c in conds:
        print(f"  {c}: {len(plan[c])} calls to run")
    print(f"estimate: ~{est_in/1e6:.2f}M input tokens, ~{est_out/1e6:.2f}M output tokens, ~${est:.2f} "
          f"at the configured prices (budget cap ${cfg['budget_usd']:.2f})")
    if args.dry_run:
        return
    if not key or key.startswith("PASTE"):
        sys.exit("No API key: put it in pipeline/config.local.json or set the environment variable.")
    if est > cfg["budget_usd"]:
        sys.exit("Estimated cost exceeds budget_usd; raise it in pipeline/config.local.json if intended.")

    spent = 0.0
    for c in conds:
        out_path = os.path.join(WORK, "outputs", f"{c}.jsonl")
        for i, r in enumerate(plan[c], 1):
            t0 = time.time()
            rec = {"id": r["id"], "cond": c, "model": cfg["model"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                resp = call(cfg, key, r["prompt"])
                u = resp.get("usage", {}) or {}
                rec.update(answer=resp["choices"][0]["message"].get("content", ""),
                           prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           cache_hit_tokens=u.get("prompt_cache_hit_tokens"))
                spent += cost(cfg, u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0,
                              u.get("prompt_cache_hit_tokens") or 0)
            except Exception as e:
                rec["error"] = str(e)[:500]
            rec["latency_s"] = round(time.time() - t0, 1)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            status = ("ERROR " + rec["error"][:120] if rec.get("error")
                      else f"ok {rec['prompt_tokens']} in / {rec['completion_tokens']} out")
            print(f"[{c} {i}/{len(plan[c])}] {r['id']}: {status} | {rec['latency_s']}s | spent ~${spent:.3f}")
            if spent > cfg["budget_usd"]:
                sys.exit("Budget cap reached; stopping. Re-run later to resume.")
    print(f"done. actual spend ~${spent:.3f} (from reported usage)")


if __name__ == "__main__":
    main()
