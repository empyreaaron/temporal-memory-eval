"""Repro: importing an old conversation into mem0 OSS writes today's date into the memory text.

The OSS SDK's add() does not accept a `timestamp` (it is platform-only), and the extraction prompt's
"Observation Date" defaults to today. So a relative phrase in an old conversation ("last week") is
resolved against today's date, and the wrong absolute date is written into the stored memory itself.
Setting created_at in metadata does not help: it changes the record's timestamp, not the extracted text.

Run (tested with mem0ai 2.1.0):
    pip install mem0ai==2.1.0
    OPENAI_API_KEY=...   python repro_import_dates.py      # mem0 defaults (OpenAI LLM + embeddings)
or  DEEPSEEK_API_KEY=... python repro_import_dates.py      # DeepSeek LLM + local embeddings (pip install fastembed==0.8.0)
The stored text is written by an LLM, so its wording, and the date in it, change with the model and the day you
run it. What it shows is that the date follows the run date, not the conversation date (2023-05-01).
"""
import datetime, logging, os, platform, sys, tempfile
from importlib import metadata

os.environ.setdefault("MEM0_TELEMETRY", "False")
from mem0 import Memory

logging.getLogger("mem0").setLevel(logging.ERROR)
print(f"Run date: {datetime.date.today()} | mem0ai {metadata.version('mem0ai')} | Python {platform.python_version()}")

path = tempfile.mkdtemp(prefix="mem0_import_")
if os.environ.get("DEEPSEEK_API_KEY"):
    m = Memory.from_config({
        "llm": {"provider": "deepseek", "config": {"model": "deepseek-flash", "temperature": 0}},
        "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant", "config": {"path": path, "embedding_model_dims": 384}},
        "history_db_path": os.path.join(path, "history.db")})
    create = m.llm.client.chat.completions.create      # mem0 expects plain JSON: turn off DeepSeek thinking mode
    m.llm.client.chat.completions.create = lambda **kw: create(**kw, extra_body={"thinking": {"type": "disabled"}})
else:
    m = Memory.from_config({"vector_store": {"provider": "qdrant", "config": {"path": path}},
                            "history_db_path": os.path.join(path, "history.db")})

# The conversation happened on 2023-05-01; we import it later, as a developer migrating history would.
m.add([{"role": "user", "content": "Last week I adopted a cat named Luna."}], user_id="import_demo",
      metadata={"created_at": "2023-05-01T09:00:00"})
print("Conversation date: 2023-05-01")
stored = m.get_all(filters={"user_id": "import_demo"})["results"]
for r in stored:
    print("  stored text:", r["memory"], "| created_at:", r.get("created_at"))
if not stored:
    sys.exit("No memory was extracted, so this run shows nothing; check the LLM settings and run it again.")
m.vector_store.client.close()
