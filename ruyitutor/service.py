from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from .auth import AuthManager
from .dify import DifyClient
from .graph import KnowledgeGraph
from .learning import LearningEngine
from .llm import GroundedLLM
from .ocr import OCRService
from .profile import ProfileStore
from .rag import LocalRAG
from .storage import Storage
from .validator import AnswerValidator


class TutorService:
    UNSAFE = ("自杀", "伤害自己", "炸弹", "毒药")
    DIFY_EVIDENCE_THRESHOLD = 0.25
    LOCAL_EVIDENCE_THRESHOLD = 6.0
    TOPIC_SPLIT_PATTERN = re.compile(
        r"请|给我|讲讲|讲解|讲|解释|说明|介绍|学习|了解|什么|怎么|如何|为什么|为何|的|和|与|及|"
        r"区别|背景|实验|一下|吗|呢"
    )

    def __init__(self, root: Path):
        self.root = root
        self.rag = LocalRAG(root / "knowledge_base" / "documents")
        self.graph = KnowledgeGraph(root / "knowledge_base" / "knowledge_graph.json")
        self.dify = DifyClient()
        self.profiles = ProfileStore()
        self.sessions: dict[str, dict] = {}
        self.validator = AnswerValidator()
        self.ocr = OCRService(self.dify)
        self.answer_logs: list[dict] = []
        self.feedback: list[dict] = []
        self.storage = Storage(root / "data" / "ruyitutor.db")
        self.learning = LearningEngine(root, self.storage, self.graph)
        self.auth = AuthManager(self.storage)
        self.llm = GroundedLLM()

    @property
    def engine_name(self) -> str:
        if self.dify.dataset_key and self.dify.dataset_id and self.llm.enabled:
            return "Dify 知识库检索 + 大模型"
        if self.llm.enabled:
            return "本地 RAG 备用 + 大模型"
        return "本地 GraphRAG 演示引擎"

    def answer(self, request: dict) -> dict:
        query = request["query"].strip()
        conversation_id = request.get("conversation_id") or uuid4().hex
        existing_session = self.sessions.get(conversation_id)
        if request.get("reveal_answer") and existing_session:
            query = existing_session["query"]
        if not request.get("reveal_answer"):
            self.storage.save_message(conversation_id, request["student_id"], "user", query)
        if any(word in query for word in self.UNSAFE):
            return {
                "answer": "我很关心你现在的安全。请立刻联系身边可信任的老师、家长或监护人；如果存在紧急危险，请拨打当地急救或报警电话。我不会提供可能造成伤害的具体方法。",
                "sources": [],
                "engine": "安全守护",
                "concepts": [],
                "confidence": 1.0,
                "conversation_id": conversation_id,
                "stage": "safety",
            }

        dify_sources = self._retrieve_dify_sources(query)
        if self._sources_are_reliable(query, dify_sources, self.DIFY_EVIDENCE_THRESHOLD):
            return self._answer_with_dify_evidence(request, query, conversation_id, dify_sources)

        local = self._retrieve_local_sources(query)
        if local:
            hits, expanded = local
            return self._answer_with_local_evidence(request, query, conversation_id, existing_session, hits, expanded)

        return self._answer_without_evidence(request, query, conversation_id)

    def _retrieve_dify_sources(self, query: str) -> list[dict]:
        if not (self.dify.dataset_key and self.dify.dataset_id):
            return []
        try:
            return self.dify.retrieve(query, top_k=5)
        except RuntimeError:
            return []

    def _retrieve_local_sources(self, query: str) -> tuple[list[dict], set[str]] | None:
        first = self.rag.search(query, top_k=3)
        if not self._local_hits_are_reliable(query, first):
            return None
        concept_ids = {item["document"].concept_id for item in first}
        expanded = self.graph.related(concept_ids, depth=2)
        hits = self.rag.search(query, top_k=4, concept_boost=expanded)
        if not self._local_hits_are_reliable(query, hits):
            return None
        return hits, expanded

    def _answer_with_dify_evidence(
        self,
        request: dict,
        query: str,
        conversation_id: str,
        sources: list[dict],
    ) -> dict:
        if self.llm.enabled:
            try:
                answer = self.llm.answer_from_sources(
                    query,
                    sources,
                    request.get("mode", "启发辅导"),
                    source_name="Dify 知识库",
                )
                engine = "Dify 知识库检索 + 大模型"
                stage = "dify-grounded"
            except Exception:
                answer = "Dify 知识库检索到了相关资料，但当前大模型调用失败。请稍后重试。"
                engine = "Dify 知识库检索成功，大模型不可用"
                stage = "llm-error"
        else:
            answer = "Dify 知识库检索到了相关资料，但当前未启用大模型。请配置 LLM_API_KEY 并设置 USE_LLM=true。"
            engine = "Dify 知识库检索成功，未启用大模型"
            stage = "llm-disabled"

        result = {
            "answer": answer,
            "sources": sources,
            "engine": engine,
            "concepts": [],
            "confidence": self._confidence(sources),
            "conversation_id": conversation_id,
            "stage": stage,
            "learning_path": [],
        }
        result["validation"] = self.validator.validate(answer, sources)
        result["validation"]["retrieval_grounded"] = bool(sources and sources[0].get("score", 0) >= self.DIFY_EVIDENCE_THRESHOLD)
        self._record_answer(request["student_id"], query, "dify-rag", result, conversation_id)
        return result

    def _answer_with_local_evidence(
        self,
        request: dict,
        query: str,
        conversation_id: str,
        existing_session: dict | None,
        hits: list[dict],
        expanded: set[str],
    ) -> dict:
        best = hits[0]["document"]
        prereqs = self.graph.prerequisites(best.concept_id)
        prereq_text = "、".join(item["name"] for item in prereqs) or "当前内容不需要额外前置知识"
        previous = existing_session
        reveal = request.get("reveal_answer", False) or (previous and previous.get("stage") == "hint")

        if self.llm.enabled:
            try:
                answer = self.llm.answer(query, hits, request.get("mode", "启发辅导"), prereq_text)
                stage = "local-grounded"
            except Exception:
                answer, stage = self._local_template_answer(request, best, prereq_text, reveal)
        else:
            answer, stage = self._local_template_answer(request, best, prereq_text, reveal)

        sources = self._sources_from_hits(hits)
        self.profiles.record(request["student_id"], best.concept_id, True)
        self.sessions[conversation_id] = {"stage": stage, "concept_id": best.concept_id, "query": query}
        result = {
            "answer": answer,
            "sources": sources,
            "engine": "本地 RAG 备用 + 大模型" if self.llm.enabled else "本地 GraphRAG",
            "concepts": [self.graph.nodes[item] for item in expanded if item in self.graph.nodes],
            "confidence": self._confidence(sources),
            "conversation_id": conversation_id,
            "stage": stage,
            "learning_path": self.graph.learning_path(best.concept_id),
        }
        result["validation"] = self.validator.validate(answer, sources)
        result["validation"]["retrieval_grounded"] = bool(sources and sources[0].get("score", 0) >= 0.5)
        if not result["validation"]["grounded"]:
            result["confidence"] = min(result["confidence"], 0.55)
        self._record_answer(request["student_id"], query, best.concept_id, result, conversation_id)
        return result

    def _answer_without_evidence(self, request: dict, query: str, conversation_id: str) -> dict:
        if self.llm.enabled:
            try:
                answer = self.llm.general_answer(
                    query,
                    request.get("mode", "启发辅导"),
                    request.get("grade", "初中"),
                    request.get("subject", "综合"),
                )
                engine = "通用大模型"
                stage = "llm-general"
                confidence = 0.42
            except Exception:
                answer = "Dify 知识库和本地 RAG 都没有可用结果，且当前大模型调用失败。请稍后重试。"
                engine = "通用大模型不可用"
                stage = "llm-error"
                confidence = 0.1
        else:
            answer = "Dify 知识库和本地 RAG 都没有可用结果，且当前未启用通用大模型。请配置 LLM_API_KEY 并设置 USE_LLM=true 后再试。"
            engine = "未启用大模型"
            stage = "llm-disabled"
            confidence = 0.1

        result = {
            "answer": answer,
            "sources": [],
            "engine": engine,
            "concepts": [],
            "confidence": confidence,
            "conversation_id": conversation_id,
            "stage": stage,
            "learning_path": [],
        }
        result["validation"] = self.validator.validate(answer, [])
        result["validation"]["retrieval_grounded"] = False
        self._record_answer(request["student_id"], query, "llm-general", result, conversation_id)
        return result

    @classmethod
    def _local_hits_are_reliable(cls, query: str, hits: list[dict]) -> bool:
        return bool(
            hits
            and hits[0]["score"] >= cls.LOCAL_EVIDENCE_THRESHOLD
            and cls._text_covers_topic(
                query,
                " ".join(f"{item['document'].title} {item['document'].content}" for item in hits),
            )
        )

    @classmethod
    def _sources_are_reliable(cls, query: str, sources: list[dict], threshold: float) -> bool:
        return bool(
            sources
            and float(sources[0].get("score", 0) or 0) >= threshold
            and cls._text_covers_topic(
                query,
                " ".join(f"{item.get('title', '')} {item.get('excerpt', '')}" for item in sources),
            )
        )

    @classmethod
    def _text_covers_topic(cls, query: str, evidence: str) -> bool:
        evidence = evidence.lower()
        terms: list[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", query.lower()):
            terms.extend(
                term for term in cls.TOPIC_SPLIT_PATTERN.split(chunk)
                if len(term) >= 2
            )
        if not terms:
            return True
        strong_terms = [term for term in terms if len(term) >= 4]
        if strong_terms:
            return any(term in evidence for term in strong_terms)
        return any(term in evidence for term in terms)

    @classmethod
    def _evidence_covers_topic(cls, query: str, hits: list[dict]) -> bool:
        return cls._local_hits_are_reliable(query, hits)

    def _local_template_answer(self, request: dict, best, prereq_text: str, reveal: bool) -> tuple[str, str]:
        body = best.content
        if request.get("mode") == "启发辅导" and not reveal:
            return (
                f"**知识定位**\n\n这个问题主要考查“{best.title}”，前置知识是：{prereq_text}。\n\n"
                f"**一步提示**\n\n{body.split('。')[0]}。先别急着看完整过程，你能告诉我下一步应该使用哪条规律吗？\n\n"
                "回复你的想法，我会根据你的回答继续提示；也可以点击“查看完整讲解”。",
                "hint",
            )
        return (
            f"**知识定位**\n\n本题考查“{best.title}”。\n\n"
            f"**分步讲解**\n\n{body}\n\n"
            "**你来试试**\n\n请用自己的话复述核心规律，并把题目中的数字换一个再完成一次。",
            "explained",
        )

    @staticmethod
    def _sources_from_hits(hits: list[dict]) -> list[dict]:
        return [{
            "title": item["document"].title,
            "source": item["document"].source,
            "score": item["score"],
            "excerpt": item["document"].content[:120].replace("\n", " "),
            "scores": {
                "lexical": item.get("lexical_score", 0),
                "semantic": item.get("semantic_score", 0),
                "graph": item.get("graph_boost", 0),
                "title": item.get("title_score", 0),
            },
        } for item in hits]

    @staticmethod
    def _confidence(sources: list[dict]) -> float:
        if not sources:
            return 0.1
        top = float(sources[0].get("score", 0))
        return round(min(0.98, 0.55 + top / (top + 5)), 2)

    def _record_answer(self, student_id: str, query: str, concept_id: str, result: dict, conversation_id: str) -> None:
        self.answer_logs.append({
            "student_id": student_id,
            "query": query,
            "concept_id": concept_id,
            "confidence": result["confidence"],
            "grounded": result["validation"]["grounded"],
        })
        self.storage.save_message(
            conversation_id,
            student_id,
            "assistant",
            result["answer"],
            {"concept_id": concept_id, "confidence": result["confidence"], "sources": result["sources"]},
        )

    def profile(self, student_id: str) -> dict:
        result = self.profiles.summary(student_id, self.graph.nodes)
        persisted = self.storage.all_mastery(student_id)
        if persisted:
            result["mastery"] = [{
                "id": x["concept_id"],
                "name": self.graph.nodes.get(x["concept_id"], {}).get("name", x["concept_id"]),
                "attempts": x["attempts"],
                "mastery": round(x["mastery"] * 100),
            } for x in persisted]
            result["overall"] = round(sum(x["mastery"] for x in result["mastery"]) / len(result["mastery"]))
            result["recommended"] = min(result["mastery"], key=lambda x: x["mastery"])
        result["learning_path"] = self.graph.learning_path(result["recommended"]["id"])
        result["plan"] = self.learning.plan(student_id)
        result["wrong_count"] = len(self.storage.wrong_book(student_id))
        return result

    def teacher_overview(self) -> dict:
        overview = self.profiles.class_summary(self.graph.nodes)
        overview.update({
            "questions": len(self.answer_logs),
            "low_confidence": sum(x["confidence"] < .6 for x in self.answer_logs),
            "feedback_count": len(self.feedback),
            "recent_questions": self.answer_logs[-8:][::-1],
        })
        return overview

    def add_feedback(self, payload: dict) -> None:
        self.feedback.append(payload)
        self.storage.add_feedback(payload)
        if payload.get("rating") == "needs_improvement":
            self.storage.quality_issue(payload.get("query", ""), payload.get("answer", ""), "teacher_feedback")

    def knowledge_entries(self) -> list[dict]:
        return json.loads((self.root / "knowledge_base" / "curriculum.json").read_text(encoding="utf-8"))

    def add_knowledge_entry(self, entry: dict) -> None:
        path = self.root / "knowledge_base" / "curriculum.json"
        entries = self.knowledge_entries()
        if any(x["id"] == entry["id"] for x in entries):
            raise ValueError("知识点 ID 已存在")
        entries.append(entry)
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        self.storage.quality_issue(entry["id"], entry["definition"], "knowledge_added")
        self.rag = LocalRAG(self.root / "knowledge_base" / "documents")
