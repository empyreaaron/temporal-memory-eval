"""Deterministic repro: mem0's search ranking ignores which memory is newer.

No LLM call is made: memories are stored with infer=False (raw text, no extraction) and each gets its
real date as created_at metadata. We then search and print the ranking. If ranking used recency, the
newer fact would come first; it is ordered by relevance score only.

Run (tested with mem0ai 2.1.0 and 2.2.1; local embeddings, no API key needed):
    pip install mem0ai==2.1.0 fastembed==0.8.0      # or: pip install -r requirements.txt
    python repro_ranking.py
Scores are printed to 6 decimals: two scores that look equal at 3 decimals are usually not tied.
The memories are stored with infer=False, so this checks search ranking only, not mem0's extraction step.
"""
import logging, os, platform, sys, tempfile
from importlib import metadata

os.environ.setdefault("MEM0_TELEMETRY", "False")
from mem0 import Memory

logging.getLogger("mem0").setLevel(logging.ERROR)
print(f"mem0ai {metadata.version('mem0ai')}, fastembed {metadata.version('fastembed')}, "
      f"Python {platform.python_version()}, embedder BAAI/bge-small-en-v1.5")

CASES = [  # (question, [(date, text), ...]); the last entry is the newest fact
    ("Where do I live?",
     [("2023-01-10", "User lives in Chicago, in an apartment near the lake."),
      ("2023-09-02", "User moved to the suburbs last month.")]),
    ("How often do I see my therapist, Dr. Smith?",
     [("2023-04-03", "User has therapy sessions with Dr. Smith every two weeks."),
      ("2023-11-03", "User now sees Dr. Smith every week.")]),
    ("How many women are on Rachel's team?",
     [("2023-01-18", "Rachel's team of 10 people has 5 women."),
      ("2023-07-20", "Rachel's team now has 6 women out of 10 people.")]),
]

path = tempfile.mkdtemp(prefix="mem0_rank_")
m = Memory.from_config({
    # The LLM is never called (infer=False); a placeholder key only satisfies client construction.
    "llm": {"provider": "openai", "config": {"model": "unused", "api_key": "unused"}},
    "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}},
    "vector_store": {"provider": "qdrant", "config": {"path": path, "embedding_model_dims": 384}},
    "history_db_path": os.path.join(path, "history.db"),
})

newest_first = 0
for i, (question, facts) in enumerate(CASES):
    uid = f"user{i}"
    for date, text in facts:
        m.add([{"role": "user", "content": text}], user_id=uid, infer=False,
              metadata={"created_at": f"{date}T09:00:00"})
    hits = m.search(question, filters={"user_id": uid}, top_k=10)["results"]
    if len(hits) != len(facts):
        sys.exit(f"expected {len(facts)} results for {question!r}, got {len(hits)}: the repro did not run as intended")
    print(f"\nQ: {question}")
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank}  score={h['score']:.6f}  created_at={(h.get('created_at') or '')[:10]}  {h['memory']}")
    newest = facts[-1][1]
    newest_first += hits[0]["memory"] == newest
print(f"\nNewest fact ranked first in {newest_first}/{len(CASES)} cases. "
      "The ranking has no recency term, so which fact comes first depends only on wording similarity.")
m.vector_store.client.close()   # close the local Qdrant store now, not during interpreter shutdown
