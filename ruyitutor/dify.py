from __future__ import annotations

import json
import os
import base64
import uuid
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import requests


class DifyClient:
    def __init__(self):
        self.base_url = os.getenv("DIFY_API_BASE_URL", "http://localhost/v1").rstrip("/")
        self.api_key = os.getenv("DIFY_APP_API_KEY", "")
        self.dataset_key = os.getenv("DIFY_DATASET_API_KEY", "")
        self.dataset_id = os.getenv("DIFY_DATASET_ID", "")
        self.user = os.getenv("DIFY_USER", "ruyitutor-demo-user")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def status(self, live: bool = False) -> dict:
        """Return secret-free configuration and optional live connectivity status."""
        result = {
            "configured": self.enabled,
            "app_key_configured": bool(self.api_key),
            "dataset_key_configured": bool(self.dataset_key),
            "dataset_id_configured": bool(self.dataset_id),
            "base_url": self.base_url,
            "ready": bool(self.api_key and self.dataset_key and self.dataset_id),
            "live": None,
        }
        if not live or not self.enabled:
            return result
        try:
            response = requests.get(f"{self.base_url}/info", headers=self._headers(), timeout=10)
            response.raise_for_status(); info = response.json()
            result["live"] = True
            result["app_name"] = info.get("name", "Dify application")
        except (requests.RequestException, ValueError) as exc:
            result["live"] = False
            result["error_type"] = type(exc).__name__
        return result

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "RuyiTutor/1.0"}

    @staticmethod
    def _request(method: str, url: str, **kwargs):
        last_error = None
        for attempt in range(3):
            try:
                return requests.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise last_error

    def chat(self, query: str, inputs: dict) -> dict:
        payload = {"inputs": inputs, "query": query, "response_mode": "blocking", "user": self.user}
        try:
            response = self._request("POST", f"{self.base_url}/chat-messages", json=payload, headers=self._headers(), timeout=90)
            response.raise_for_status(); result = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Dify 暂时不可用：{exc}") from exc
        resources = result.get("metadata", {}).get("retriever_resources", [])
        resources = [item for item in resources if float(item.get("score", 0) or 0) >= 0.25]
        return {
            "answer": result.get("answer", ""),
            "sources": [{
                "title": item.get("document_name", "知识库资料"),
                "score": round(float(item.get("score", 0)), 3),
                "excerpt": item.get("content", "")[:420],
            } for item in resources],
            "engine": "Dify RAG",
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.dataset_key or not self.dataset_id:
            return []
        body = {
            "query": query,
            "retrieval_model": {
                "search_method": "hybrid_search", "reranking_enable": True,
                "top_k": top_k, "score_threshold_enabled": True, "score_threshold": 0.25,
            },
        }
        try:
            headers=self._headers(); headers["Authorization"]=f"Bearer {self.dataset_key}"
            response=self._request("POST",f"{self.base_url}/datasets/{self.dataset_id}/retrieve",json=body,headers=headers,timeout=30)
            response.raise_for_status(); result=response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Dify 检索不可用：{exc}") from exc
        return [{
            "title": r.get("segment", {}).get("document", {}).get("name", "知识库资料"),
            "score": round(float(r.get("score", 0)), 4),
            "excerpt": r.get("segment", {}).get("content", "")[:220],
        } for r in result.get("records", [])]

    def vision_ocr(self, image_base64: str) -> str:
        if not self.enabled:
            return ""
        header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("data:image/png;base64", image_base64)
        mime = header.split(";")[0].replace("data:", "") or "image/png"
        extension = mime.split("/")[-1].replace("jpeg", "jpg")
        raw = base64.b64decode(encoded)
        try:
            headers={"Authorization":f"Bearer {self.api_key}","User-Agent":"RuyiTutor/1.0"}
            response=requests.post(f"{self.base_url}/files/upload",headers=headers,data={"user":self.user},files={"file":(f"question.{extension}",raw,mime)},timeout=60)
            response.raise_for_status(); file_id=response.json()["id"]
            payload = {"inputs":{},"query":"只提取图片中的题干、公式和选项，保持原顺序，不要解题。","response_mode":"blocking","user":self.user,
                       "files":[{"type":"image","transfer_method":"local_file","upload_file_id":file_id}]}
            response=requests.post(f"{self.base_url}/chat-messages",json=payload,headers=self._headers(),timeout=90)
            response.raise_for_status(); return str(response.json().get("answer", "")).strip()
        except (requests.RequestException, KeyError, ValueError):
            return ""
