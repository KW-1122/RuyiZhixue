from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
from .vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    text = text.lower()
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    stop = {"什么", "怎么", "如何", "为什么", "是不是", "是否", "应该", "可以", "一样", "这个", "那个"}
    bigrams = ["".join(chinese[i : i + 2]) for i in range(len(chinese) - 1)]
    bigrams = [token for token in bigrams if token not in stop]
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    return [word for word in words if word not in stop] + bigrams

ALIASES={"移项":"方程 等式 两边运算","斜率":"一次函数 k","快慢":"速度","浮起来":"浮力","轻重":"质量","锋利":"压强 受力面积","开口":"二次函数 抛物线","验根":"分式方程 增根"}

def rewrite_query(query: str) -> str:
    additions=[value for key,value in ALIASES.items() if key in query]
    return query+(" "+" ".join(additions) if additions else "")


@dataclass
class Document:
    id: str
    title: str
    subject: str
    grade: str
    concept_id: str
    source: str
    content: str


class LocalRAG:
    """Small dependency-free BM25-style retriever for offline demonstration."""

    def __init__(self, knowledge_dir: Path):
        self.documents = self._load(knowledge_dir)
        self.documents.extend(self._load_curriculum(knowledge_dir.parent / "curriculum.json"))
        self.documents.extend(self._load_curriculum(knowledge_dir.parent / "expanded" / "expanded_catalog.json"))
        self.documents.extend(self._load_public_chunks(knowledge_dir.parent / "public_curriculum_chunks.json"))
        self.doc_tokens = [tokenize(d.title + " " + d.content) for d in self.documents]
        self.df = Counter(token for tokens in self.doc_tokens for token in set(tokens))
        self.avg_len = sum(map(len, self.doc_tokens)) / max(len(self.doc_tokens), 1)
        self.vector_store = VectorStore(self.documents, knowledge_dir.parents[1])

    def _load_public_chunks(self, path: Path) -> list[Document]:
        if not path.exists():
            return []
        return [Document(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

    def _load(self, folder: Path) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(folder.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            parts = raw.split("---", 2)
            if len(parts) != 3:
                continue
            meta = {}
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            docs.append(Document(
                id=path.stem,
                title=meta.get("title", path.stem),
                subject=meta.get("subject", "综合"),
                grade=meta.get("grade", "初中"),
                concept_id=meta.get("concept_id", path.stem),
                source=meta.get("source", "如意智学团队自编知识条目"),
                content=parts[2].strip(),
            ))
        return docs

    def _load_curriculum(self, path: Path) -> list[Document]:
        if not path.exists():
            return []
        entries = json.loads(path.read_text(encoding="utf-8"))
        docs: list[Document] = []
        for item in entries:
            common = dict(
                subject=item["subject"], grade=item["grade"],
                concept_id=item["id"], source=item["source"],
            )
            variants = [
                ("概念", item["definition"]),
                ("方法与公式", item.get("method", "")),
                ("例题", item["example"]),
                ("易错点", item["mistake"]),
                ("练习", item["practice"]),
            ]
            for kind, content in variants:
                if not content:
                    continue
                docs.append(Document(
                    id=f'{item["id"]}_{kind}', title=f'{item["name"]}·{kind}',
                    content=content, **common,
                ))
        return docs

    def search(self, query: str, top_k: int = 4, concept_boost: set[str] | None = None) -> list[dict]:
        query=rewrite_query(query)
        vector_scores = self.vector_store.scores(query)
        q_tokens = tokenize(query)
        q_count = Counter(q_tokens)
        results = []
        for doc_index, (doc, tokens) in enumerate(zip(self.documents, self.doc_tokens)):
            tf = Counter(tokens)
            lexical = 0.0
            for token, qf in q_count.items():
                if token not in tf:
                    continue
                idf = math.log(1 + (len(self.documents) - self.df[token] + 0.5) / (self.df[token] + 0.5))
                freq = tf[token]
                norm = freq + 1.5 * (0.25 + 0.75 * len(tokens) / max(self.avg_len, 1))
                lexical += qf * idf * (freq * 2.5 / norm)
            weighted_q={t:c*math.log(1+len(self.documents)/(1+self.df[t])) for t,c in q_count.items()}
            weighted_d={t:c*math.log(1+len(self.documents)/(1+self.df[t])) for t,c in tf.items() if t in q_count}
            dot=sum(weighted_q.get(t,0)*v for t,v in weighted_d.items())
            qnorm=math.sqrt(sum(v*v for v in weighted_q.values()));dnorm=math.sqrt(sum(v*v for v in weighted_d.values()))
            semantic=dot/(qnorm*dnorm) if qnorm and dnorm else 0.0
            title_overlap=len(set(tokenize(doc.title))&set(q_tokens))/max(len(set(tokenize(doc.title))),1)
            graph_bonus=1.2 if concept_boost and doc.concept_id in concept_boost else 0.0
            vector_score=max(0.0, vector_scores.get(doc_index, 0.0))
            score=lexical+semantic*2.0+vector_score*4.0+title_overlap*2.0+graph_bonus
            if score > 0:
                results.append({"document": doc, "score": round(score, 4),"lexical_score":round(lexical,4),"semantic_score":round(vector_score if vector_scores else semantic,4),"graph_boost":graph_bonus,"title_score":round(title_overlap,4),"vector_backend":"BGE-small-zh-v1.5 + FAISS" if vector_scores else "char-tfidf fallback"})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
