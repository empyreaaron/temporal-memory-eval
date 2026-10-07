"""mem0 runs: how the open-source mem0 memory layer handles facts that change. Run prepare_data.py first.

Two tasks, all answers from the same model and the same answer settings:
  fc   MemoryAgentBench FactConsolidation (numbered facts, newer fact = larger number).
       FULL (all facts in the prompt), BM25 (top-20 facts), MEM0 (facts ingested into mem0),
       MEM0_r2 (MEM0 answered again, for noise), MEM0_T (same store; memories sorted by mem0's created_at,
       labeled with one number per distinct created_at value, and a rule saying larger numbers are newer;
       a label is a timestamp group, not necessarily one add() batch), MEM0_S (re-ingested with the
       extractor told to keep one memory per fact, not merge, and keep each fact's serial number).
  lme  LongMemEval-S questions listed in data/question_ids.json.
       M0   plain default mem0: nothing about dates at write time, relevance order, no dates shown
       Mnd  write-time dates (session date told to the extractor, stored as created_at); relevance order;
            no dates shown
       Mord as Mnd but listed oldest to newest (and the prompt says so); no dates shown
       M    oldest-to-newest order with each memory's date shown; M_r2/M_r3 repeat M for noise
       Mord and M (the published conditions) sort by calendar day only: memories from the same day keep
       mem0's relevance order. Mord_ts and M_ts are the same conditions sorted by the full timestamp; they
       are not run unless requested with --variants (supplementary results, 2026-10-07).
       M0_wo  the plain default store, listed by mem0's own created_at (the write order) with the "oldest to
            newest" line; no dates shown. Run with --default-variant M0_wo (supplementary results, 2026-10-07).
       run_baselines.py supplies F/R/O for the same questions.

Usage (from the repository root):
  python pipeline/run_mem0.py --dry-run                      # token/cost estimate only, no key needed
  python pipeline/run_mem0.py --smoke                        # tiny end-to-end test (~$0.03), outputs_smoke/
  python pipeline/run_mem0.py --task fc --fc-conds FULL,BM25,MEM0,MEM0_r2,MEM0_T,MEM0_S
  python pipeline/run_mem0.py --task lme --variants M,Mnd,Mord,M_r2,M_r3 --default-usage all
  python pipeline/run_mem0.py --task lme --lme-types multi-session,temporal-reasoning --variants M,Mnd,Mord --default-usage all
Re-run any command to resume after an interruption. The last record per question decides its status (the
graders use the same rule): finished items are skipped, failed ones are retried. Every record carries a
fingerprint of the settings, prompts and inputs that produced it; if an output file or memory store was
made with different settings, the run stops instead of mixing them (use a new TME_WORK directory).
Each run appends its settings (no key), package versions and input hashes to outputs/run_manifest.jsonl.
The exit code is non-zero unless every planned item finished.

Settings: pipeline/config.example.json, overridden by pipeline/config.local.json (key goes there or in the
environment variable named by "api_key_env"). Never prints the key. Never reads gold/.
Work directory (inputs/, outputs/, store/, models/): ./work, or set TME_WORK.
mem0 telemetry is switched off; embeddings run locally on CPU (downloaded once into <work>/models/).
budget_usd is checked before each API call, so calls already in flight can take the spend slightly past it.
"""
import argparse, collections, hashlib, json, math, os, platform, re, shutil, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata

PIPE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("TME_WORK", os.path.join(os.path.dirname(PIPE), "work"))
os.environ["MEM0_TELEMETRY"] = "False"
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(WORK, "models"))

FC_RULE = ("Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a "
           "serial number at the beginning, and the newer fact has a larger serial number. You need to solve the "
           "conflicts of facts in the knowledge pool by finding the newest fact with the larger serial number. "
           "Answer the question based on this rule, only from the knowledge pool rather than real-world facts. "
           "Give a very concise answer without other words.")          # adapted from MemoryAgentBench
FC_PROMPT = "{rule}\n\nKnowledge pool:\n{pool}\n\nQuestion: {question}\nAnswer:"
FC_CHUNK_MSG = "Here is a list of facts to remember:\n{facts}"
# MEM0_T: order recovered from mem0's own created_at timestamps (facts were written batch by batch, in order).
# One label per distinct created_at value: one add() batch can span several timestamps (in the published runs
# 10 batches gave 17 labels at 6k, 47 batches gave 84 at 32k), so labels mark write order, not batch identity.
FC_RULE_T = ("Pretend you are a knowledge management system. Each memory in the knowledge pool is labeled with the "
             "order in which it was stored; a memory with a larger order number was stored later and is newer, and "
             "memories with the same number were stored at the same time. You "
             "need to solve the conflicts of memories by finding the newest one. Answer the question based on this "
             "rule, only from the knowledge pool rather than real-world facts. Give a very concise answer without "
             "other words.")
# MEM0_S: ask mem0's extractor to keep each fact's serial number inside the memory text
FC_SERIAL_NOTE = ("Each fact in the message starts with a serial number. Extract exactly one memory per fact, do not "
                  "merge facts, and keep that fact's serial number at the start of the memory in the form "
                  "'[#<number>] <fact>'.")
FC_MEM0_STORE = {"MEM0": "base", "MEM0_r2": "base", "MEM0_T": "base", "MEM0_S": "serial"}
LME_PROMPT = ("I will give you memories about a user, extracted from past chats between you and the user; each "
              "memory has the date of the chat it came from. Please answer the question based on the relevant "
              "memories. If the memories do not contain enough information, say so.\n\nMemories:\n\n{memories}"
              "\n\nCurrent Date: {date}\nQuestion: {question}\nAnswer (concise):")
LME_DATE_NOTE = ("These messages were exchanged on {date}. Treat that as the Observation Date: use it to resolve "
                 "relative time references and to date the memories.")   # OSS mem0 has no timestamp input

PRINT_LOCK, CTX, STOP = threading.Lock(), threading.local(), threading.Event()


def log(msg):
    with PRINT_LOCK:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_config():
    """config.example.json supplies defaults; config.local.json overrides them (so an older local
    config keeps working when new settings are added)."""
    with open(os.path.join(PIPE, "config.example.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    path = os.path.join(PIPE, "config.local.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    key = os.environ.get(cfg.get("api_key_env", ""), "") or cfg.get("api_key", "")
    return cfg, key, os.path.basename(path if os.path.exists(path) else "config.example.json")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rec):
    with PRINT_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def latest_by_id(path):
    """Last record per id (re-runs append). The runner, grader and exporter all use this rule."""
    return {x["id"]: x for x in load_jsonl(path)}


# ---------------------------------------------------------------- fingerprints and run manifest
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def answer_settings(cfg):
    return {k: cfg.get(k) for k in ("model", "base_url", "answer_thinking", "reasoning_effort")}


def store_settings(cfg, kind):
    """Everything that shapes a memory store, except the per-item input (checked separately)."""
    s = {"kind": kind, "model": cfg["model"], "base_url": cfg["base_url"], "embed_model": cfg["embed_model"],
         "mem0ai": pkg_version("mem0ai")}
    if kind in ("base", "serial"):
        s.update(chunk_facts=cfg["fc_chunk_facts"], chunk_msg=FC_CHUNK_MSG,
                 extract_note=FC_SERIAL_NOTE if kind == "serial" else None)
    else:
        s.update(extract_note=LME_DATE_NOTE if kind == "patched" else None)
    return s


DEFAULT_VARIANTS = {"M0": ("relevance", False), "M0_wo": ("time", False)}   # answer variants of the default store


def variant_def(cfg, v):
    order = ("time" if v in cfg.get("lme_variants_time_order", []) else
             "date" if v in cfg["lme_variants_date_order"] else "relevance")
    return order, v in cfg["lme_variants_with_dates"]


def output_fingerprints(cfg, inputs_sha):
    """One fingerprint per output file: answer settings, prompt, retrieval settings, store settings, inputs."""
    base = {"answer": answer_settings(cfg)}
    fps = {}
    for c in cfg["fc_conditions"]:
        spec = dict(base, prompt=FC_PROMPT, rule=FC_RULE_T if c == "MEM0_T" else FC_RULE, inputs=inputs_sha.get("fc"))
        if c == "BM25":
            spec["bm25_k"] = cfg["fc_bm25_k"]
        elif c in FC_MEM0_STORE:
            spec.update(top_k=cfg["mem0_top_k"], store=store_settings(cfg, FC_MEM0_STORE[c]))
        fps[f"fc_{c}.jsonl"] = digest(spec)
    for v, mode in [(v, "patched") for v in cfg["lme_variants"]] + [(cfg["lme_default_variant"], "default")]:
        order, dates = variant_def(cfg, v) if mode == "patched" else DEFAULT_VARIANTS[v]
        fps[f"{v}.jsonl"] = digest(dict(base, prompt=LME_PROMPT, order=order, dates_shown=dates,
                                        top_k=cfg["mem0_top_k"], store=store_settings(cfg, mode),
                                        inputs=inputs_sha.get("lme")))
    return fps


def check_store(store, settings, item_input):
    """A finished store is reused only if it was built with the same settings and the same input."""
    want = {"settings": settings, "input": digest(item_input)}
    path = os.path.join(store, "STORE_CONFIG.json")
    have = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
    if have != want:
        raise RuntimeError(f"memory store {os.path.basename(store)} was built with other settings or input "
                           "(or by an older version of this script); delete it or use a new work directory")


def write_store_config(store, settings, item_input):
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, "STORE_CONFIG.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "input": digest(item_input)}, f, indent=1)


def write_manifest(out_dir, entry):
    append_jsonl(os.path.join(out_dir, "run_manifest.jsonl"), entry)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(PIPE), capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- cost meter and API wrapper
class BudgetExceeded(Exception):
    pass


class Meter:
    def __init__(self, cfg, path):
        self.cfg, self.path, self.lock, self.spent, self.calls = cfg, path, threading.Lock(), 0.0, 0

    def cost(self, u):
        c, tin, tout = self.cfg, u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        hit = u.get("prompt_cache_hit_tokens") or 0
        return ((tin - hit) * c["price_in_per_m"] + hit * c["price_in_cache_hit_per_m"]
                + tout * c["price_out_per_m"]) / 1e6

    def check(self):
        if STOP.is_set():
            raise BudgetExceeded("stopped")
        if self.spent > self.cfg["budget_usd"]:
            STOP.set()
            raise BudgetExceeded(f"budget cap ${self.cfg['budget_usd']:.2f} reached")

    def record(self, kind, usage, latency, out_chars, model=None):
        """latency_s is the time of this one API call (the successful attempt), nothing else. Answer calls also
        log the call_id, fingerprint and run of the answer record they produced, so a summary can keep exactly
        the calls behind the graded answers."""
        c = self.cost(usage)
        with self.lock:
            self.spent += c
            self.calls += 1
        append_jsonl(self.path, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
                                 "item": getattr(CTX, "item", "?"), "latency_s": round(latency, 1),
                                 "out_chars": out_chars, "cost": round(c, 6), "model": model,
                                 "run": self.cfg.get("_run"), **(getattr(CTX, "call", None) or {}),
                                 **{k: usage.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                              "prompt_cache_hit_tokens")},
                                 "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")})


FATAL_STATUS = {400, 401, 402, 403, 404, 422}   # bad request / key / balance: retrying will not help


def metered(create, meter, kind, extra_body=None):
    def wrapped(**kw):
        if extra_body is not None:
            kw["extra_body"] = {**(kw.get("extra_body") or {}), **extra_body}
        delay = 5
        for attempt in range(6):
            meter.check()
            t0 = time.time()
            try:
                r = create(**kw)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in FATAL_STATUS or attempt == 5:
                    if status in (401, 402, 403):
                        STOP.set()
                        log(f"FATAL HTTP {status} (key or balance problem): {str(e)[:200]}")
                    raise
                log(f"    {kind}: {type(e).__name__} {status or ''} - retry in {delay}s")
                time.sleep(delay)
                delay *= 2
                continue
            u = r.usage.model_dump() if getattr(r, "usage", None) else {}
            meter.record(kind, u, time.time() - t0, len(r.choices[0].message.content or ""),
                         getattr(r, "model", None))                 # model id as reported by the API
            return r
    return wrapped


# ---------------------------------------------------------------- mem0 setup
def be_gentle(cfg):
    """Keep the machine usable: run at below-normal priority and cap the local embedder's CPU threads
    (onnxruntime otherwise grabs every core)."""
    import functools
    import fastembed
    import mem0.embeddings.fastembed as mfe
    mfe.TextEmbedding = functools.partial(fastembed.TextEmbedding, threads=cfg.get("embed_threads", 2))
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000):          # BELOW_NORMAL_PRIORITY_CLASS
            log("note: could not lower process priority; the run continues at normal priority")


def patch_mem0():
    """Quiet mem0 (no remote notice fetch, no console notices) and share one CPU embedder across threads."""
    import logging
    import mem0.memory.main as mm
    import mem0.memory.notices as mn
    from mem0.utils.factory import EmbedderFactory
    logging.getLogger("mem0").setLevel(logging.ERROR)
    mn._fetch_remote_config = lambda: None
    for name in list(vars(mm)):
        if name.startswith("display_") and name.endswith("_notice"):
            setattr(mm, name, lambda *a, **k: None)
    shared, lock, orig = {}, threading.RLock(), EmbedderFactory.create.__func__   # embed_batch may call embed

    def locked(fn):
        def inner(*a, **k):
            with lock:
                return fn(*a, **k)
        return inner

    def create(cls, provider_name, config, vector_config):
        with lock:
            if provider_name not in shared:
                e = orig(cls, provider_name, config, vector_config)
                e.embed, e.embed_batch = locked(e.embed), locked(e.embed_batch)
                shared[provider_name] = e
            return shared[provider_name]
    EmbedderFactory.create = classmethod(create)


def make_memory(cfg, key, store, meter):
    from mem0 import Memory
    os.makedirs(store, exist_ok=True)
    m = Memory.from_config({
        "llm": {"provider": "deepseek", "config": {"model": cfg["model"], "api_key": key,
                                                   "deepseek_base_url": cfg["base_url"], "max_tokens": 8192}},
        "embedder": {"provider": "fastembed", "config": {"model": cfg["embed_model"], "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "mem", "path": store,
                                                          "embedding_model_dims": 384, "on_disk": True}},
        "history_db_path": os.path.join(store, "history.db"),
    })
    # mem0's own calls (memory extraction) run with thinking off, as mem0 expects a plain JSON reply.
    m.llm.client.chat.completions.create = metered(m.llm.client.chat.completions.create, meter, "mem0_extract",
                                                   extra_body={"thinking": {"type": "disabled"}})
    return m


def dump_memories(m, user_id, path):
    got = m.get_all(filters={"user_id": user_id}, top_k=100000)
    rows = [{"id": x["id"], "text": x["memory"], "created_at": x.get("created_at"),
             "metadata": x.get("metadata")} for x in got.get("results", [])]
    rows.sort(key=lambda x: x["created_at"] or "")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return len(rows)


# ---------------------------------------------------------------- helpers
tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())


def bm25_top(query, docs, k, k1=1.5, b=0.75):                  # same BM25 as the baseline R prompts
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
    return sorted(range(n), key=lambda i: -scores[i])[:k]


def lme_iso(date_str):                                           # "2023/05/20 (Sat) 02:21" -> ISO
    m = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", date_str)
    return datetime(*map(int, m.groups())).isoformat() if m else None


def fmt_date(iso):
    return iso[:10].replace("-", "/") if iso else "unknown date"


def time_key(iso):
    """Full created_at as a comparable time, or None if missing or unparseable. Times with a UTC offset
    (mem0's own timestamps) are converted to UTC; times without one (the session dates we store) are taken
    as given. One store holds only one kind, so the two are never compared with each other."""
    try:
        t = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def order_memories(by_score, order):
    """'relevance': mem0's order. 'date': the published Mord/M ordering, by calendar day only, so memories
    from the same day keep relevance order (and an unknown date sorts last). 'time': by full timestamp;
    memories without a usable timestamp are put after the dated ones, in relevance order."""
    if order == "relevance":
        return list(by_score)
    if order == "date":
        return sorted(by_score, key=lambda x: x["date"])
    dated = [x for x in by_score if time_key(x.get("created_at"))]
    undated = [x for x in by_score if not time_key(x.get("created_at"))]
    return sorted(dated, key=lambda x: time_key(x["created_at"])) + undated


def answer_fn(cfg, key, meter):
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=cfg["timeout_s"], max_retries=0)
    create = metered(client.chat.completions.create, meter, "answer")

    def answer(prompt, rec):
        """rec: the output record this answer goes into; its call_id and fp are written to the call log."""
        kw = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
        if cfg["answer_thinking"]:
            kw.update(extra_body={"thinking": {"type": "enabled"}}, reasoning_effort=cfg["reasoning_effort"])
        else:
            kw.update(extra_body={"thinking": {"type": "disabled"}}, temperature=0)
        CTX.call = {"call_id": rec.get("call_id"), "fp": rec.get("fp")}
        try:
            return create(**kw).choices[0].message.content or ""
        finally:
            CTX.call = None                                   # mem0's own extraction calls carry no call_id
    return answer


# ---------------------------------------------------------------- plan
def build_plan(cfg, smoke):
    fc = [r for r in load_jsonl(os.path.join(WORK, "inputs", "fc.jsonl")) if r["source"] in cfg["fc_sources"]]
    lme = [r for r in load_jsonl(os.path.join(WORK, "inputs", "lme.jsonl")) if r["type"] in cfg["lme_types"]]
    if cfg.get("lme_skip_b_stage", True):
        lme = [r for r in lme if not r.get("in_b_stage")]     # those were used while developing the method
    lme.sort(key=lambda r: (not r.get("in_c1"), r["id"]))     # stable order
    if smoke:
        fc = [dict(fc[0], facts=fc[0]["facts"][: 2 * cfg["fc_chunk_facts"]], questions=fc[0]["questions"][:2])] if fc else []
        lme = [dict(lme[0], sessions=lme[0]["sessions"][:3])] if lme else []
    usage = cfg.get("lme_default_usage")
    default = lme if usage == "all" else [r for r in lme if r.get("in_c1")] if usage == "in_c1" else \
              lme[: int(usage or 0)]
    return fc, lme, default


def estimate(cfg, fc, lme, default=(), ingested=frozenset()):
    """Rough token estimate (chars/4). mem0's ~8.4k-token system prompt is assumed to hit DeepSeek's cache."""
    t = lambda s: len(s) / 4
    miss = hit = out = 0.0
    n_ans = 0
    for r in fc:
        pool = "\n".join(r["facts"])
        nq = len(r["questions"])
        stores = set()
        for cond in cfg["fc_conditions"]:
            out += nq * 200                                   # fc answers are a few words (~115 tokens measured)
            if cond == "FULL":
                miss += t(pool) + nq * 60; hit += (nq - 1) * t(pool)
            elif cond == "BM25":
                miss += nq * (t(FC_RULE) + 20 * 16)
            elif cond in FC_MEM0_STORE:
                stores.add(FC_MEM0_STORE[cond])
                miss += nq * (t(FC_RULE) + cfg["mem0_top_k"] * 25)
        for kind in stores:                                   # ingest only stores not built yet
            if fc_store_name(r["source"], kind) in ingested:
                continue
            chunks = [r["facts"][i:i + cfg["fc_chunk_facts"]] for i in range(0, len(r["facts"]), cfg["fc_chunk_facts"])]
            for i, c in enumerate(chunks):                    # last 10 chunks are resent as "Last k Messages"
                miss += t("\n".join(c)) * (1 + min(i, 10)) + 800; hit += 8400; out += len(c) * 30
    for r, n_v, pre in [(r, len(cfg["lme_variants"]), "lme") for r in lme] + [(r, 1, "lme0") for r in default]:
        for s in ([] if f"{pre}_{r['id']}" in ingested else r["sessions"]):   # stored memories are reused
            miss += sum(t(x["content"]) for x in s["turns"]) + 1000; hit += 7600; out += 750   # per-call overheads measured in our runs
        miss += n_v * (cfg["mem0_top_k"] * 40 + 200)
        n_ans += n_v
    out += n_ans * cfg["est_answer_out_tokens"]
    c = cfg
    usd = (miss * c["price_in_per_m"] + hit * c["price_in_cache_hit_per_m"] + out * c["price_out_per_m"]) / 1e6
    return miss, hit, out, usd


# ---------------------------------------------------------------- tasks
def new_rec(cfg, out_file, **kw):
    """Every output record carries the settings fingerprint of its file, the id of the run that wrote it, and a
    call_id that the call log repeats for the API call behind the answer."""
    return {**kw, "fp": cfg["_fp"].get(out_file), "run": cfg.get("_run"), "call_id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def fc_simple(cfg, answer, row, q, cond, out_path):
    CTX.item = f"fc_{cond}:{q['id']}"
    if cond == "FULL":
        pool = "\n".join(row["facts"])
    else:
        idx = sorted(bm25_top(q["question"], row["facts"], cfg["fc_bm25_k"]))
        pool = "\n".join(row["facts"][i] for i in idx)
    rec = new_rec(cfg, f"fc_{cond}.jsonl", id=q["id"], source=row["source"], cond=cond)
    try:
        rec["answer"] = answer(FC_PROMPT.format(rule=FC_RULE, pool=pool, question=q["question"]), rec)
    except Exception as e:
        rec["error"] = str(e)[:300]
    append_jsonl(out_path, rec)
    return rec


def fc_store_name(source, kind):
    return f"fc_{source}" if kind == "base" else f"fc_S_{source}"


def fc_mem0_ingest(cfg, key, meter, row, store_root, dump_dir, kind="base"):
    """kind 'base': plain mem0 (first run). kind 'serial': mem0 asked to keep each fact's serial number."""
    name = fc_store_name(row["source"], kind)
    CTX.item = f"{name}_ingest"
    store = os.path.join(store_root, name)
    marker = os.path.join(store, "INGEST_DONE")
    settings = store_settings(cfg, kind)
    if os.path.exists(marker):
        check_store(store, settings, row["facts"])
        return make_memory(cfg, key, store, meter)
    shutil.rmtree(store, ignore_errors=True)
    m = make_memory(cfg, key, store, meter)
    write_store_config(store, settings, row["facts"])
    n = cfg["fc_chunk_facts"]
    chunks = [row["facts"][i:i + n] for i in range(0, len(row["facts"]), n)]
    for i, c in enumerate(chunks, 1):
        m.add([{"role": "user", "content": FC_CHUNK_MSG.format(facts="\n".join(c))}], user_id="fc",
              prompt=FC_SERIAL_NOTE if kind == "serial" else None)
        log(f"  [{name}] ingested chunk {i}/{len(chunks)} | spent ~${meter.spent:.3f}")
    count = dump_memories(m, "fc", os.path.join(dump_dir, f"{name}.json"))
    with open(marker, "w") as f:
        f.write(str(count))
    log(f"  [{name}] {len(row['facts'])} facts -> {count} memories")
    return m


def fc_mem0_question(cfg, answer, m, mlock, row, q, out_path, cond="MEM0"):
    CTX.item = f"fc_{cond}:{q['id']}"
    rec = new_rec(cfg, f"fc_{cond}.jsonl", id=q["id"], source=row["source"], cond=cond)
    try:
        with mlock:
            hits = m.search(q["question"], top_k=cfg["mem0_top_k"], filters={"user_id": "fc"})["results"]
        if cond == "MEM0_T":        # one label per distinct created_at value; equal timestamps share a label
            hits.sort(key=lambda h: h.get("created_at") or "")
            rank = {t: i for i, t in enumerate(sorted({h.get("created_at") or "" for h in hits}), 1)}
            rec["retrieved"] = [{"order": rank[h.get("created_at") or ""], "created_at": h.get("created_at"),
                                 "text": h["memory"]} for h in hits]
            pool = "\n".join(f"(stored {x['order']}) {x['text']}" for x in rec["retrieved"])
            rule = FC_RULE_T
        else:
            rec["retrieved"] = [h["memory"] for h in hits]
            pool, rule = "\n".join(rec["retrieved"]), FC_RULE
        rec["answer"] = answer(FC_PROMPT.format(rule=rule, pool=pool, question=q["question"]), rec)
    except Exception as e:
        rec["error"] = str(e)[:300]
    append_jsonl(out_path, rec)
    return rec


def lme_item(cfg, key, meter, answer, r, store_root, dump_dir, out_dir, mode, variants):
    """mode 'patched': tell mem0 each session's date and store it as the memory date (our workaround).
       mode 'default': plain mem0 as a developer gets it out of the box (no dates at all).
       Variants of one store differ only in the order of the memories and whether their dates are shown."""
    qid = r["id"]
    prefix = "lme" if mode == "patched" else "lme0"
    CTX.item = f"{prefix}:{qid}"
    store = os.path.join(store_root, f"{prefix}_{qid}")
    marker = os.path.join(store, "INGEST_DONE")
    settings = store_settings(cfg, mode)
    recs = []
    try:
        if os.path.exists(marker):
            check_store(store, settings, r["sessions"])
            m = make_memory(cfg, key, store, meter)
        else:
            shutil.rmtree(store, ignore_errors=True)
            m = make_memory(cfg, key, store, meter)
            write_store_config(store, settings, r["sessions"])
            for i, s in enumerate(r["sessions"], 1):
                msgs = [{"role": t["role"], "content": t["content"]} for t in s["turns"] if t["content"].strip()]
                if msgs:
                    if mode == "patched":
                        m.add(msgs, user_id=qid, metadata={"session_date": s["date"], "created_at": lme_iso(s["date"])},
                              prompt=LME_DATE_NOTE.format(date=s["date"]))
                    else:
                        m.add(msgs, user_id=qid)
                if i % 10 == 0 or i == len(r["sessions"]):
                    log(f"  [{prefix} {qid}] session {i}/{len(r['sessions'])} | spent ~${meter.spent:.3f}")
            with open(marker, "w") as f:
                f.write(str(dump_memories(m, qid, os.path.join(dump_dir, f"{prefix}_{qid}.json"))))
        hits = m.search(r["question"], top_k=cfg["mem0_top_k"], filters={"user_id": qid})["results"]
        by_score = [{"date": fmt_date(h.get("created_at")), "created_at": h.get("created_at"), "text": h["memory"],
                     "score": round(h.get("score") or 0, 3)} for h in hits]          # mem0's own order
        with open(marker) as f:
            n_mem = int(f.read())
        for v in variants:
            # Two separate switches. Order: oldest to newest, or mem0's relevance order, which leaks no
            # recency information. Display: show each memory's date or not. Mord = date order, no dates,
            # with one line saying the list runs oldest to newest (otherwise the model cannot know).
            order, show_dates = variant_def(cfg, v) if mode == "patched" else DEFAULT_VARIANTS[v]
            retrieved = order_memories(by_score, order)
            n_undated = sum(1 for x in retrieved if not time_key(x["created_at"]))
            rec = new_rec(cfg, f"{v}.jsonl", id=qid, cond=v, mode=mode, type=r["type"], n_memories=n_mem,
                          order=order, dates_shown=show_dates, n_undated=n_undated, retrieved=retrieved)
            mems = "\n".join((f"- ({x['date']}) {x['text']}" if show_dates else f"- {x['text']}") for x in retrieved)
            if order == "time" and n_undated:
                mems = "(Listed from oldest to newest; memories without a known date come last.)\n" + mems
                log(f"  [{prefix} {qid} {v}] note: {n_undated} retrieved memories have no usable timestamp")
            elif order != "relevance" and not show_dates:
                mems = "(Listed from oldest to newest.)\n" + mems
            CTX.item = f"{prefix}:{qid}:{v}"                  # tag the call log with the answer variant
            try:
                rec["answer"] = answer(LME_PROMPT.format(memories=mems, date=r["question_date"], question=r["question"]), rec)
            except Exception as e:
                rec["error"] = str(e)[:300]
            append_jsonl(os.path.join(out_dir, f"{v}.jsonl"), rec)
            recs.append(rec)
    except Exception as e:
        for v in variants[len(recs):]:                       # variants not already recorded above
            rec = new_rec(cfg, f"{v}.jsonl", id=qid, cond=v, mode=mode, type=r["type"], error=str(e)[:300])
            append_jsonl(os.path.join(out_dir, f"{v}.jsonl"), rec)
            recs.append(rec)
    return recs[0] if recs else {"id": qid, "cond": variants[0], "error": "no variant ran"}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", choices=["fc", "lme", "all"], default="all")
    ap.add_argument("--fc-conds", help="override fc_conditions, comma-separated, e.g. MEM0_r2,MEM0_T,MEM0_S")
    ap.add_argument("--lme-types", help="override lme_types, e.g. multi-session,temporal-reasoning")
    ap.add_argument("--variants", help="override lme_variants (patched-store answer variants), e.g. M,Mnd,Mord")
    ap.add_argument("--default-usage", help="which questions also get the plain-default M0 run: all, in_c1 or a count")
    ap.add_argument("--default-variant", choices=sorted(DEFAULT_VARIANTS),
                    help="answer variant on the default store: M0 (relevance order) or M0_wo (write order, no dates)")
    args = ap.parse_args()
    cfg, key, cfg_name = load_config()
    if args.fc_conds:
        cfg["fc_conditions"] = args.fc_conds.split(",")
    if args.lme_types:
        cfg["lme_types"] = args.lme_types.split(",")
    if args.variants:
        cfg["lme_variants"] = args.variants.split(",")
    if args.default_usage:
        cfg["lme_default_usage"] = args.default_usage
    if args.default_variant:
        cfg["lme_default_variant"] = args.default_variant
    bad = [c for c in cfg["fc_conditions"] if c not in ("FULL", "BM25", *FC_MEM0_STORE)]
    if bad:
        sys.exit(f"Unknown fc condition(s): {bad}")
    known = {"Mnd", *cfg["lme_variants_date_order"], *cfg.get("lme_variants_time_order", []),
             *cfg["lme_variants_with_dates"]}
    bad = [v for v in cfg["lme_variants"] if v not in known]
    if bad:     # an unlisted name would silently run as relevance order without dates
        sys.exit(f"Unknown lme variant(s): {bad}; known: {sorted(known)}")
    fc, lme, default = build_plan(cfg, args.smoke)
    if args.task == "fc": lme, default = [], []
    if args.task == "lme": fc = []

    out_dir = os.path.join(WORK, "outputs_smoke" if args.smoke else "outputs")
    store_root = os.path.join(WORK, "store_smoke" if args.smoke else "store")
    if args.smoke and not args.dry_run:
        shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree(store_root, ignore_errors=True)
    dump_dir = os.path.join(out_dir, "mem_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    out_files = [f"fc_{c}.jsonl" for c in cfg["fc_conditions"]] + [f"{v}.jsonl" for v in cfg["lme_variants"]] \
                + [f"{cfg['lme_default_variant']}.jsonl"]
    inputs = {k: os.path.join(WORK, "inputs", f"{k}.jsonl") for k in ("fc", "lme")}
    inputs_sha = {k: file_sha256(p) for k, p in inputs.items() if os.path.exists(p)}
    cfg["_fp"] = output_fingerprints(cfg, inputs_sha)
    cfg["_run"] = time.strftime("%Y%m%d-%H%M%S")
    done, foreign = {}, {}
    for p in out_files:              # the last record per id decides: finished items are skipped, errors retried
        last = latest_by_id(os.path.join(out_dir, p))
        done[p] = {i for i, x in last.items() if not x.get("error")}
        foreign[p] = sorted(i for i, x in last.items() if not x.get("error") and x.get("fp") != cfg["_fp"][p])
    clash = {p: ids for p, ids in foreign.items() if ids}
    if clash:
        sys.exit("These output files hold answers made with other settings, prompts or inputs (or by an older "
                 "version of this script), so resuming would mix conditions:\n"
                 + "\n".join(f"  {p}: {len(ids)} answers, e.g. {ids[:3]}" for p, ids in clash.items())
                 + f"\nUse a new work directory (set TME_WORK) or move those files out of {out_dir}.")
    fc = [dict(r, questions=qs) for r in fc                      # drop items finished in an earlier run
          if (qs := [q for q in r["questions"]
                     if any(q["id"] not in done[f"fc_{c}.jsonl"] for c in cfg["fc_conditions"])])]
    lme = [r for r in lme if any(r["id"] not in done[f"{v}.jsonl"] for v in cfg["lme_variants"])]
    default = [r for r in default if r["id"] not in done[f"{cfg['lme_default_variant']}.jsonl"]]
    planned = ({(f"fc_{c}.jsonl", q["id"]) for r in fc for q in r["questions"] for c in cfg["fc_conditions"]}
               | {(f"{v}.jsonl", r["id"]) for r in lme for v in cfg["lme_variants"]}
               | {(f"{cfg['lme_default_variant']}.jsonl", r["id"]) for r in default})
    planned = {(p, i) for p, i in planned if i not in done[p]}
    ingested = {d for d in (os.listdir(store_root) if os.path.isdir(store_root) else [])
                if os.path.exists(os.path.join(store_root, d, "INGEST_DONE"))}
    miss, hit, out, usd = estimate(cfg, fc, lme, default, ingested)
    print(f"config: {cfg_name} | model: {cfg['model']} | base_url: {cfg['base_url']}")
    print(f"fc: {[r['source'] for r in fc]} x {cfg['fc_conditions']}")
    print(f"lme: {len(lme)} questions {cfg['lme_types']} x {cfg['lme_variants']} | "
          f"default-usage run on {len(default)} of them as {cfg['lme_default_variant']}")
    print(f"answers to produce: {len(planned)}")
    print(f"estimate (remaining work only): ~{miss/1e6:.2f}M uncached + ~{hit/1e6:.2f}M cached "
          f"input tokens, ~{out/1e6:.2f}M output tokens, ~${usd:.2f} at peak prices (budget cap ${cfg['budget_usd']:.2f})")
    if args.dry_run or not planned:
        return
    if not key or key.startswith("PASTE"):
        sys.exit("No API key: put it in pipeline/config.local.json or set the environment variable.")
    if usd > cfg["budget_usd"]:
        sys.exit("Estimated cost exceeds budget_usd; raise it in pipeline/config.local.json if intended.")
    write_manifest(out_dir, {
        "run": cfg["_run"], "event": "start", "argv": sys.argv[1:], "config_file": cfg_name,
        "settings": {k: v for k, v in cfg.items() if k not in ("api_key", "_fp", "_run")},
        "fingerprints": {p: cfg["_fp"][p] for p in out_files}, "inputs_sha256": inputs_sha,
        "python": sys.version.split()[0], "platform": platform.platform(), "git_commit": git_commit(),
        "packages": {p: pkg_version(p) for p in ("mem0ai", "openai", "fastembed", "qdrant-client", "onnxruntime")},
        "planned_answers": len(planned)})

    be_gentle(cfg)
    patch_mem0()
    meter = Meter(cfg, os.path.join(out_dir, "calls.jsonl"))
    answer = answer_fn(cfg, key, meter)
    t0 = time.time()
    # Warm DeepSeek's prefix cache: FULL prompts share one long prefix per source, so run one first.
    if "FULL" in cfg["fc_conditions"]:
        for r in fc:
            q = next((q for q in r["questions"] if q["id"] not in done["fc_FULL.jsonl"]), None)
            if q:
                fc_simple(cfg, answer, r, q, "FULL", os.path.join(out_dir, "fc_FULL.jsonl"))
                done["fc_FULL.jsonl"].add(q["id"])
                log(f"  [fc FULL {r['source']}] cache warm-up call done | spent ~${meter.spent:.3f}")
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futs = {}
        mem0_conds = [c for c in cfg["fc_conditions"] if c in FC_MEM0_STORE]
        for r in fc:                                           # long mem0 ingest jobs first
            kinds = {FC_MEM0_STORE[c] for c in mem0_conds
                     if any(q["id"] not in done[f"fc_{c}.jsonl"] for q in r["questions"])}
            for kind in sorted(kinds):
                futs[pool.submit(fc_mem0_ingest, cfg, key, meter, r, store_root, dump_dir, kind)] = ("ingest", (r, kind))
        for r in lme:
            todo = [v for v in cfg["lme_variants"] if r["id"] not in done[f"{v}.jsonl"]]
            if todo:
                futs[pool.submit(lme_item, cfg, key, meter, answer, r, store_root, dump_dir, out_dir,
                                 "patched", todo)] = ("lme", r)
        for r in default:
            v = cfg["lme_default_variant"]
            if r["id"] not in done[f"{v}.jsonl"]:
                futs[pool.submit(lme_item, cfg, key, meter, answer, r, store_root, dump_dir, out_dir,
                                 "default", [v])] = ("lme0", r)
        for r in fc:
            for cond in [c for c in cfg["fc_conditions"] if c not in FC_MEM0_STORE]:
                for q in r["questions"]:
                    if q["id"] not in done[f"fc_{cond}.jsonl"]:
                        futs[pool.submit(fc_simple, cfg, answer, r, q, cond,
                                         os.path.join(out_dir, f"fc_{cond}.jsonl"))] = ("fc", q)
        pending = set(futs)
        while pending:
            for f in as_completed(list(pending)):
                pending.discard(f)
                kind, r = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    log(f"ERROR in {kind} job: {str(e)[:200]}")
                    if kind == "ingest":                       # record the failure for every question it blocks
                        row, store_kind = r
                        for c in [c for c in mem0_conds if FC_MEM0_STORE[c] == store_kind]:
                            for q in row["questions"]:
                                if q["id"] not in done[f"fc_{c}.jsonl"]:
                                    append_jsonl(os.path.join(out_dir, f"fc_{c}.jsonl"),
                                                 new_rec(cfg, f"fc_{c}.jsonl", id=q["id"], source=row["source"],
                                                         cond=c, error="ingest failed: " + str(e)[:280]))
                    continue
                if kind == "ingest":                           # then answer that store's questions
                    row, store_kind = r
                    mlock = threading.Lock()
                    for c in [c for c in mem0_conds if FC_MEM0_STORE[c] == store_kind]:
                        for q in row["questions"]:
                            if q["id"] not in done[f"fc_{c}.jsonl"]:
                                nf = pool.submit(fc_mem0_question, cfg, answer, res, mlock, row, q,
                                                 os.path.join(out_dir, f"fc_{c}.jsonl"), c)
                                futs[nf] = ("fc", q); pending.add(nf)
                    break
                label = r["id"]
                status = "ERROR " + res["error"][:100] if res.get("error") else "ok"
                if kind in ("lme", "lme0") or res.get("error"):
                    log(f"[{kind} {res.get('cond')}] {label}: {status} | spent ~${meter.spent:.3f}")
    sys.exit(finish(cfg, out_dir, out_files, planned, meter, t0))


def run_status(cfg, out_dir, out_files, planned):
    """Compare the plan with what this run wrote: ok, failed, or not run at all."""
    last = {p: latest_by_id(os.path.join(out_dir, p)) for p in out_files}
    counts = collections.Counter()
    for p, i in planned:
        x = last[p].get(i)
        counts["not run" if not x or x.get("run") != cfg["_run"] else "failed" if x.get("error") else "ok"] += 1
    return counts


def finish(cfg, out_dir, out_files, planned, meter, t0):
    counts = run_status(cfg, out_dir, out_files, planned)
    complete = counts["ok"] == len(planned)
    write_manifest(out_dir, {"run": cfg["_run"], "event": "end", "ok": counts["ok"], "failed": counts["failed"],
                             "not_run": counts["not run"], "api_calls": meter.calls, "spent_usd": round(meter.spent, 4)})
    print(f"{'done' if complete else 'INCOMPLETE'} in {(time.time()-t0)/60:.1f} min | {meter.calls} API calls | "
          f"spent ~${meter.spent:.3f} | planned {len(planned)}: {counts['ok']} ok, {counts['failed']} failed, "
          f"{counts['not run']} not run")
    if STOP.is_set():
        print("Stopped early (budget cap or key/balance problem).")
    if not complete:
        print("Re-run the same command to retry the failed and missing items. Failed runs are not wrong answers: "
              "do not grade until this command reports done.")
    return 0 if complete else 1


if __name__ == "__main__":
    main()
