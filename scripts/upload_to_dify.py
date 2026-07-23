from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
import argparse
from pathlib import Path
from urllib.parse import urlencode
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
BASE = os.getenv("DIFY_API_BASE_URL", "http://localhost/v1").rstrip("/")
KEY = os.getenv("DIFY_DATASET_API_KEY", "")
NAME = os.getenv("DIFY_DATASET_NAME", "如意智学-初中知识库")


def api(method: str, path: str, data: bytes | None = None, content_type: str = "application/json") -> dict:
    headers = {"Authorization": f"Bearer {KEY}", "Accept": "application/json", "Content-Type": content_type,
               "User-Agent": "RuyiTutor-DifySync/1.0 (+https://cloud.dify.ai)"}
    for attempt in range(1, 5):
        try:
            response = requests.request(method, BASE + path, data=data, headers=headers, timeout=180)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                if attempt == 4:
                    raise RuntimeError("Dify知识库请求频率限制在多次退避后仍未解除")
                time.sleep(attempt * 20)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Dify API {response.status_code}: {response.text[:500]}")
            return response.json() if response.content else {}
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Dify网络连接连续失败4次：{type(exc).__name__}") from exc
            time.sleep(attempt * 2)
    return {}


def dataset_id() -> str:
    result = api("GET", "/datasets?" + urlencode({"page": 1, "limit": 100, "keyword": NAME}))
    for item in result.get("data", []):
        if item.get("name") == NAME:
            return item["id"]
    body = json.dumps({"name": NAME, "indexing_technique": "high_quality"}, ensure_ascii=False).encode()
    return api("POST", "/datasets", body)["id"]


def existing_names(ds_id: str) -> set[str]:
    names: set[str] = set()
    page = 1
    while True:
        result = api("GET", f"/datasets/{ds_id}/documents?" + urlencode({"page": page, "limit": 100}))
        items = result.get("data", [])
        names.update(str(item.get("name", "")) for item in items)
        if not result.get("has_more") or not items:
            return names
        page += 1


def upload(ds_id: str, path: Path) -> None:
    boundary = "----RuyiTutor" + uuid.uuid4().hex
    meta = json.dumps({"indexing_technique": "high_quality", "process_rule": {"mode": "automatic"}}).encode()
    mime = mimetypes.guess_type(path.name)[0] or "text/markdown"
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"data\"\r\n\r\n".encode(), meta, b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        path.read_bytes(), b"\r\n", f"--{boundary}--\r\n".encode(),
    ])
    result = api("POST", f"/datasets/{ds_id}/document/create-by-file", body, f"multipart/form-data; boundary={boundary}")
    print(f"已上传 {path.name}，batch={result.get('batch', '')}")


def upload_text(ds_id: str, item: dict, kind: str, content: str) -> None:
    name = f'{item["grade"]}-{item["subject"]}-{item["name"]}-{kind}'
    text = f'# {item["name"]}·{kind}\n\n年级：{item["grade"]}\n学科：{item["subject"]}\n知识点ID：{item["id"]}\n来源：{item["source"]}\n\n{content}'
    body = json.dumps({"name": name, "text": text, "indexing_technique": "high_quality", "doc_form": "text_model", "process_rule": {"mode": "automatic"}}, ensure_ascii=False).encode("utf-8")
    result = api("POST", f"/datasets/{ds_id}/document/create-by-text", body)
    print(f"已上传 {name}，batch={result.get('batch', '')}")


def upload_concept(ds_id: str, item: dict) -> None:
    name = f'{item["grade"]}-{item["subject"]}-{item["name"]}'
    text = (
        f'# {item["name"]}\n\n年级：{item["grade"]}\n学科：{item["subject"]}\n'
        f'知识点ID：{item["id"]}\n来源：{item["source"]}\n\n'
        f'## 概念\n\n{item["definition"]}\n\n'
        f'## 例题\n\n{item["example"]}\n\n'
        f'## 易错点\n\n{item["mistake"]}\n\n'
        f'## 练习\n\n{item["practice"]}'
    )
    body = json.dumps({"name": name, "text": text, "indexing_technique": "high_quality", "doc_form": "text_model", "process_rule": {"mode": "automatic"}}, ensure_ascii=False).encode("utf-8")
    result = api("POST", f"/datasets/{ds_id}/document/create-by-text", body)
    print(f"已上传合并知识点 {name}，batch={result.get('batch', '')}")


def delete_api_documents(ds_id: str) -> int:
    result = api("GET", f"/datasets/{ds_id}/documents?" + urlencode({"page": 1, "limit": 100}))
    targets = [item for item in result.get("data", []) if item.get("created_from") == "api"]
    for item in targets:
        api("DELETE", f'/datasets/{ds_id}/documents/{item["id"]}')
        print(f'已删除旧文档 {item.get("name", item["id"])}')
        time.sleep(1.2)
    return len(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="幂等同步如意智学知识到Dify")
    parser.add_argument("--dry-run", action="store_true", help="只显示待上传条目")
    parser.add_argument("--force", action="store_true", help="即使同名文档存在也再次上传")
    parser.add_argument("--replace-consolidated", action="store_true", help="删除API创建的旧文档并按25文档结构重传")
    parser.add_argument("--replace-expanded", action="store_true", help="删除API创建的旧文档并上传审核后的33个扩充章节文档")
    parser.add_argument("--expanded", action="store_true", help="幂等续传审核后的33个扩充章节文档，不删除现有文档")
    args = parser.parse_args()
    if not KEY:
        raise SystemExit("请先设置 DIFY_DATASET_API_KEY")
    ds = dataset_id()
    deleted = delete_api_documents(ds) if ((args.replace_consolidated or args.replace_expanded) and not args.dry_run) else 0
    known = existing_names(ds)
    uploaded = skipped = 0
    source_dir = ROOT / "knowledge_base" / "expanded" / "dify_documents" if (args.replace_expanded or args.expanded) else ROOT / "knowledge_base" / "documents"
    for file in sorted(source_dir.glob("*.md")):
        if file.name in known and not args.force:
            skipped += 1; continue
        if not args.dry_run: upload(ds, file)
        uploaded += 1
        time.sleep(1.5)
    curriculum = json.loads((ROOT / "knowledge_base" / "curriculum.json").read_text(encoding="utf-8"))
    if args.replace_expanded or args.expanded:
        pass
    elif args.replace_consolidated:
        for item in curriculum:
            name = f'{item["grade"]}-{item["subject"]}-{item["name"]}'
            if name in known and not args.force:
                skipped += 1; continue
            if not args.dry_run: upload_concept(ds, item)
            uploaded += 1; time.sleep(1.5)
    else:
        fields = {"概念": "definition", "例题": "example", "易错点": "mistake", "练习": "practice"}
        for item in curriculum:
            for kind, field in fields.items():
                name = f'{item["grade"]}-{item["subject"]}-{item["name"]}-{kind}'
                if name in known and not args.force:
                    skipped += 1; continue
                if not args.dry_run: upload_text(ds, item, kind, item[field])
                uploaded += 1; time.sleep(1.5)
    logical_chunks = 350 if (args.replace_expanded or args.expanded) else 79
    print(json.dumps({"dataset": NAME, "dataset_id": ds, "deleted": deleted, "uploaded": uploaded, "skipped": skipped, "logical_chunks": logical_chunks, "dry_run": args.dry_run}, ensure_ascii=False))
