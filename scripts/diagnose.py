from __future__ import annotations
import importlib.util, json, os, socket, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
checks = {
    "python": sys.version.split()[0],
    "fastapi": bool(importlib.util.find_spec("fastapi")),
    "uvicorn": bool(importlib.util.find_spec("uvicorn")),
    "knowledge_chunks": None,
    "database_writable": os.access(ROOT / "data", os.W_OK) if (ROOT / "data").exists() else os.access(ROOT, os.W_OK),
    "dify_app_key": bool(os.getenv("DIFY_APP_API_KEY")),
    "dify_dataset_key": bool(os.getenv("DIFY_DATASET_API_KEY")),
    "dify_dataset_id": bool(os.getenv("DIFY_DATASET_ID")),
}
sys.path.insert(0, str(ROOT))
from ruyitutor.rag import LocalRAG
checks["knowledge_chunks"] = len(LocalRAG(ROOT / "knowledge_base" / "documents").documents)
port = int(os.getenv("PORT", "8000"))
with socket.socket() as sock:
    checks["port_available"] = sock.connect_ex(("127.0.0.1", port)) != 0
print(json.dumps(checks, ensure_ascii=False, indent=2))
required = checks["fastapi"] and checks["uvicorn"] and checks["knowledge_chunks"] >= 50 and checks["database_writable"]
raise SystemExit(0 if required else 1)
