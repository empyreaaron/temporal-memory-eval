"""mem0 runs: how the open-source mem0 memory layer handles facts that change. Run prepare_data.py first.

Two tasks, all answers from the same model and the same answer settings:
  fc   MemoryAgentBench FactConsolidation (numbered facts, newer fact = larger number).
       FULL (all facts in the prompt), BM25 (top-20 facts), MEM0 (facts ingested into mem0),
       MEM0_r2 (MEM0 answered again, for noise), MEM0_T (same store, memories labeled by storage batch from
       mem0's created_at), MEM0_S (re-ingested with the extractor told to keep one memory per fact, not merge,
       and keep each fact's serial number).
  lme  LongMemEval-S questions listed in data/question_ids.json.
       M0   plain default mem0: nothing about dates at write time, relevance order, no dates shown
       Mnd  write-time dates (session date told to the extractor, stored as created_at); relevance order;
            no dates shown
       Mord as Mnd but listed oldest to newest (and the prompt says so); no dates shown
       M    chronological order with each memory's date shown; M_r2/M_r3 repeat M for noise
       run_baselines.py supplies F/R/O for the same questions.

Usage (from the repository root):
  python pipeline/run_mem0.py --dry-run                      # token/cost estimate only, no key needed
  python pipeline/run_mem0.py --smoke                        # tiny end-to-end test (~$0.03), outputs_smoke/
  python pipeline/run_mem0.py --task fc --fc-conds FULL,BM25,MEM0,MEM0_r2,MEM0_T,MEM0_S
  python pipeline/run_mem0.py --task lme --variants M,Mnd,Mord,M_r2,M_r3 --default-usage all
  python pipeline/run_mem0.py --task lme --lme-types multi-session,temporal-reasoning                               --variants M,Mnd,Mord --default-usage all
Re-run any command to resume after an interruption; finished items are skipped.

Settings: pipeline/config.example.json, overridden by pipeline/config.local.json (key goes there or in the
environment variable named by "api_key_env"). Never prints the key. Never reads gold/.
Work directory (inputs/, outputs/, store/, models/): ./work, or set TME_WORK.
mem0 telemetry is switched off; embeddings run locally on CPU (downloaded once into <work>/models/).
"""
import argparse, collections, json, math, os, re, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
# MEM0_T: order recovered from mem0's own created_at timestamps (facts were written batch by batch, in order)
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

    def record(self, kind, usage, latency, out_chars):
        c = self.cost(usage)
        with self.lock:
            self.spent += c
            self.calls += 1
        append_jsonl(self.path, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
                                 "item": getattr(CTX, "item", "?"), "latency_s": round(latency, 1),
                                 "out_chars": out_chars, "cost": round(c, 6),
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
            meter.record(kind, u, time.time() - t0, len(r.choices[0].message.content or ""))
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


def answer_fn(cfg, key, meter):
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=cfg["timeout_s"], max_retries=0)
    create = metered(client.chat.completions.create, meter, "answer")

    def answer(prompt):
        kw = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
        if cfg["answer_thinking"]:
            kw.update(extra_body={"thinking": {"type": "enabled"}}, reasoning_effort=cfg["reasoning_effort"])
        else:
            kw.update(extra_body={"thinking": {"type": "disabled"}}, temperature=0)
        return create(**kw).choices[0].message.content or ""
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
def fc_simple(cfg, answer, row, q, cond, out_path):
    CTX.item = f"fc_{cond}:{q['id']}"
    if cond == "FULL":
        pool = "\n".join(row["facts"])
    else:
        idx = sorted(bm25_top(q["question"], row["facts"], cfg["fc_bm25_k"]))
        pool = "\n".join(row["facts"][i] for i in idx)
    rec = {"id": q["id"], "source": row["source"], "cond": cond, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        rec["answer"] = answer(FC_PROMPT.format(rule=FC_RULE, pool=pool, question=q["question"]))
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
    if os.path.exists(marker):
        return make_memory(cfg, key, store, meter)
    shutil.rmtree(store, ignore_errors=True)
    m = make_memory(cfg, key, store, meter)
    n = cfg["fc_chunk_facts"]
    chunks = [row["facts"][i:i + n] for i in range(0, len(row["facts"]), n)]
    for i, c in enumerate(chunks, 1):
        m.add([{"role": "user", "content": FC_CHUNK_MSG.format(facts="\n".join(c))}], user_id="fc",
              prompt=FC_SERIAL_NOTE if kind == "serial" else None)
        log(f"  [{name}] ingested chunk {i}/{len(chunks)} | spent ~${meter.spent:.3f}")
    count = dump_memories(m, "fc", os.path.join(dump_dir, f"{name}.json"))
    open(marker, "w").write(str(count))
    log(f"  [{name}] {len(row['facts'])} facts -> {count} memories")
    return m


def fc_mem0_question(cfg, answer, m, mlock, row, q, out_path, cond="MEM0"):
    CTX.item = f"fc_{cond}:{q['id']}"
    rec = {"id": q["id"], "source": row["source"], "cond": cond, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with mlock:
            hits = m.search(q["question"], top_k=cfg["mem0_top_k"], filters={"user_id": "fc"})["results"]
        if cond == "MEM0_T":        # label by storage batch (mem0's created_at); equal timestamps share a label
            hits.sort(key=lambda h: h.get("created_at") or "")
            rank = {t: i for i, t in enumerate(sorted({h.get("created_at") or "" for h in hits}), 1)}
            rec["retrieved"] = [{"order": rank[h.get("created_at") or ""], "created_at": h.get("created_at"),
                                 "text": h["memory"]} for h in hits]
            pool = "\n".join(f"(stored {x['order']}) {x['text']}" for x in rec["retrieved"])
            rule = FC_RULE_T
        else:
            rec["retrieved"] = [h["memory"] for h in hits]
            pool, rule = "\n".join(rec["retrieved"]), FC_RULE
        rec["answer"] = answer(FC_PROMPT.format(rule=rule, pool=pool, question=q["question"]))
    except Exception as e:
        rec["error"] = str(e)[:300]
    append_jsonl(out_path, rec)
    return rec


def lme_item(cfg, key, meter, answer, r, store_root, dump_dir, out_dir, mode, variants):
    """mode 'patched': tell mem0 each session's date and store it as the memory date (our workaround).
       mode 'default': plain mem0 as a developer gets it out of the box (no dates at all).
       Variants differ only in whether the answer prompt shows memory dates."""
    qid = r["id"]
    prefix = "lme" if mode == "patched" else "lme0"
    CTX.item = f"{prefix}:{qid}"
    store = os.path.join(store_root, f"{prefix}_{qid}")
    marker = os.path.join(store, "INGEST_DONE")
    recs = []
    try:
        if os.path.exists(marker):
            m = make_memory(cfg, key, store, meter)
        else:
            shutil.rmtree(store, ignore_errors=True)
            m = make_memory(cfg, key, store, meter)
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
            open(marker, "w").write(str(dump_memories(m, qid, os.path.join(dump_dir, f"{prefix}_{qid}.json"))))
        hits = m.search(r["question"], top_k=cfg["mem0_top_k"], filters={"user_id": qid})["results"]
        by_score = [{"date": fmt_date(h.get("created_at")), "text": h["memory"],
                     "score": round(h.get("score") or 0, 3)} for h in hits]          # mem0's own order
        by_date = sorted(by_score, key=lambda x: x["date"])
        n_mem = int(open(marker).read())
        for v in variants:
            # Two separate switches. Order: chronological (date order) or mem0's relevance order, which leaks
            # no recency information. Display: show each memory's date or not. Mord = date order, no dates,
            # with one line saying the list runs oldest to newest (otherwise the model cannot know).
            date_order = v in cfg["lme_variants_date_order"]
            show_dates = v in cfg["lme_variants_with_dates"]
            retrieved = by_date if date_order else by_score
            rec = {"id": qid, "cond": v, "mode": mode, "type": r["type"], "n_memories": n_mem,
                   "order": "date" if date_order else "relevance", "dates_shown": show_dates,
                   "retrieved": retrieved, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            mems = "\n".join((f"- ({x['date']}) {x['text']}" if show_dates else f"- {x['text']}") for x in retrieved)
            if date_order and not show_dates:
                mems = "(Listed from oldest to newest.)\n" + mems
            CTX.item = f"{prefix}:{qid}:{v}"                  # tag the call log with the answer variant
            try:
                rec["answer"] = answer(LME_PROMPT.format(memories=mems, date=r["question_date"], question=r["question"]))
            except Exception as e:
                rec["error"] = str(e)[:300]
            append_jsonl(os.path.join(out_dir, f"{v}.jsonl"), rec)
            recs.append(rec)
    except Exception as e:
        for v in variants:
            rec = {"id": qid, "cond": v, "mode": mode, "type": r["type"], "error": str(e)[:300],
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
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
    bad = [c for c in cfg["fc_conditions"] if c not in ("FULL", "BM25", *FC_MEM0_STORE)]
    if bad:
        sys.exit(f"Unknown fc condition(s): {bad}")
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
    done = {p: {x["id"] for x in load_jsonl(os.path.join(out_dir, p)) if not x.get("error")} for p in out_files}
    fc = [dict(r, questions=qs) for r in fc                      # drop items finished in an earlier run
          if (qs := [q for q in r["questions"]
                     if any(q["id"] not in done[f"fc_{c}.jsonl"] for c in cfg["fc_conditions"])])]
    lme = [r for r in lme if any(r["id"] not in done[f"{v}.jsonl"] for v in cfg["lme_variants"])]
    default = [r for r in default if r["id"] not in done[f"{cfg['lme_default_variant']}.jsonl"]]
    ingested = {d for d in (os.listdir(store_root) if os.path.isdir(store_root) else [])
                if os.path.exists(os.path.join(store_root, d, "INGEST_DONE"))}
    miss, hit, out, usd = estimate(cfg, fc, lme, default, ingested)
    print(f"config: {cfg_name} | model: {cfg['model']} | base_url: {cfg['base_url']}")
    print(f"fc: {[r['source'] for r in fc]} x {cfg['fc_conditions']}")
    print(f"lme: {len(lme)} questions {cfg['lme_types']} x {cfg['lme_variants']} | "
          f"default-usage run on {len(default)} of them as {cfg['lme_default_variant']}")
    print(f"estimate (remaining work only): ~{miss/1e6:.2f}M uncached + ~{hit/1e6:.2f}M cached "
          f"input tokens, ~{out/1e6:.2f}M output tokens, ~${usd:.2f} at peak prices (budget cap ${cfg['budget_usd']:.2f})")
    if args.dry_run:
        return
    if not key or key.startswith("PASTE"):
        sys.exit("No API key: put it in pipeline/config.local.json or set the environment variable.")
    if usd > cfg["budget_usd"]:
        sys.exit("Estimated cost exceeds budget_usd; raise it in pipeline/config.local.json if intended.")

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
    latest = {}                                                # last record per (file, id) decides its status
    for p in out_files:
        for x in load_jsonl(os.path.join(out_dir, p)):
            latest[(p, x["id"])] = x
    n_err = sum(1 for x in latest.values() if x.get("error"))
    print(f"done in {(time.time()-t0)/60:.1f} min | {meter.calls} API calls | spent ~${meter.spent:.3f}"
          + (f" | {n_err} error rows (re-run to retry them)" if n_err else ""))
    if STOP.is_set():
        print("Stopped early (budget cap or key/balance problem). Re-run to resume.")


if __name__ == "__main__":
    main()
