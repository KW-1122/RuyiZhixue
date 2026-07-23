from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json
import os
import re

from .dify import DifyClient
from .graph import KnowledgeGraph
from .profile import ProfileStore
from .rag import LocalRAG
from .validator import AnswerValidator
from .ocr import OCRService
from .storage import Storage
from .learning import LearningEngine
from .auth import AuthManager
from .llm import GroundedLLM


class TutorService:
    UNSAFE = ("自杀", "伤害自己", "炸弹", "毒药")
    NAMED_TOPIC_PATTERNS = (
        re.compile(r"[一二三四五六七八九十两]+元[一二三四五六七八九十两]+次方程组"),
        re.compile(r"[一二三四五六七八九十两]+元[一二三四五六七八九十两]+次不等式组"),
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
        if self.dify.enabled and os.getenv("RAG_PIPELINE_MODE", "local").lower() == "dify": return "Dify RAG（本地降级已就绪）"
        if self.llm.enabled: return "DeepSeek + BGE/FAISS GraphRAG"
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
                "sources": [], "engine": "安全守护", "concepts": [], "confidence": 1.0,
                "conversation_id": conversation_id, "stage": "safety",
            }
        if self.dify.enabled and os.getenv("RAG_PIPELINE_MODE", "local").lower() == "dify":
            try:
                result = self.dify.chat(query, {k: request[k] for k in ("grade", "subject", "mode")})
                result.update({"concepts": [], "confidence": self._confidence(result["sources"]),
                               "conversation_id": conversation_id, "stage": "guided", "learning_path": []})
                result["validation"] = self.validator.validate(result["answer"], result["sources"])
                result["validation"]["retrieval_grounded"] = bool(result["sources"] and result["sources"][0].get("score", 0) >= 0.5)
                self.answer_logs.append({"student_id": request["student_id"], "query": query, "concept_id": "dify-rag", "confidence": result["confidence"], "grounded": result["validation"]["grounded"]})
                return result
            except RuntimeError:
                pass
        first = self.rag.search(query, top_k=3)
        if not first or first[0]["score"] < 6.0 or not self._evidence_covers_named_topic(query, first):
            return {
                "answer": "这个问题超出了当前初中数学与物理知识库的可靠范围。我不想凭空猜测。你可以换一种说法，或请老师把对应资料加入知识库后再问我。",
                "sources": [], "engine": "本地 GraphRAG", "concepts": [], "confidence": 0.12,
                "conversation_id": conversation_id, "stage": "refused",
            }
        concept_ids = {item["document"].concept_id for item in first}
        expanded = self.graph.related(concept_ids, depth=2)
        hits = self.rag.search(query, top_k=4, concept_boost=expanded)
        # Empirically calibrated on the bundled in-domain/out-of-domain set.
        # A low lexical match must not be promoted into a confident teaching answer.
        if not hits or hits[0]["score"] < 6.0:
            return {
                "answer": "这个问题超出了当前初中数学与物理知识库的可靠范围。我不想凭空猜测。你可以换一种说法，或请老师把对应资料加入知识库后再问我。",
                "sources": [], "engine": "本地 GraphRAG", "concepts": [], "confidence": 0.12,
                "conversation_id": conversation_id, "stage": "refused",
            }
        best = hits[0]["document"]
        prereqs = self.graph.prerequisites(best.concept_id)
        body = best.content
        prereq_text = "、".join(item["name"] for item in prereqs) or "当前内容不需要额外前置知识"
        previous = existing_session
        reveal = request.get("reveal_answer", False) or (previous and previous.get("stage") == "hint")
        if request.get("mode") == "启发辅导" and not reveal:
            answer = (
                f"**知识定位**\n\n这个问题主要考查「{best.title}」，前置知识是：{prereq_text}。\n\n"
                f"**一步提示**\n\n{body.split('。')[0]}。先别急着看完整过程，你能告诉我下一步应该使用哪个规律吗？\n\n"
                "回复你的想法，我会根据你的回答继续提示；也可以点击“查看完整讲解”。"
            )
            stage = "hint"
        else:
            answer = (
                f"**知识定位**\n\n本题考查「{best.title}」。\n\n"
                f"**分步讲解**\n\n{body}\n\n"
                "**你来试试**\n\n请用自己的话复述核心规律，并把题目中的数字换一个再完成一次。"
            )
            stage = "explained"
        if self.llm.enabled:
            try:
                answer = self.llm.answer(query, hits, request.get("mode", "启发辅导"), prereq_text)
                stage = "llm-grounded"
            except Exception:
                pass
        sources = [{
            "title": item["document"].title,
            "source": item["document"].source,
            "score": item["score"],
            "excerpt": item["document"].content[:120].replace("\n", " "),
            "scores": {"lexical":item.get("lexical_score",0),"semantic":item.get("semantic_score",0),"graph":item.get("graph_boost",0),"title":item.get("title_score",0)},
        } for item in hits]
        self.profiles.record(request["student_id"], best.concept_id, True)
        self.sessions[conversation_id] = {"stage": stage, "concept_id": best.concept_id, "query": query}
        result = {
            "answer": answer,
            "sources": sources,
            "engine": "DeepSeek + BGE/FAISS GraphRAG" if self.llm.enabled else "本地 GraphRAG",
            "concepts": [self.graph.nodes[item] for item in expanded if item in self.graph.nodes],
            "confidence": self._confidence(sources),
            "conversation_id": conversation_id,
            "stage": stage,
            "learning_path": self.graph.learning_path(best.concept_id),
        }
        result["validation"] = self.validator.validate(answer, sources)
        if not result["validation"]["grounded"]:
            result["confidence"] = min(result["confidence"], 0.55)
        self.answer_logs.append({"student_id": request["student_id"], "query": query, "concept_id": best.concept_id, "confidence": result["confidence"], "grounded": result["validation"]["grounded"]})
        self.storage.save_message(conversation_id, request["student_id"], "assistant", answer, {"concept_id":best.concept_id,"confidence":result["confidence"],"sources":sources})
        return result

    @classmethod
    def _evidence_covers_named_topic(cls, query: str, hits: list[dict]) -> bool:
        evidence = " ".join(
            f"{item['document'].title} {item['document'].content}" for item in hits
        )
        for pattern in cls.NAMED_TOPIC_PATTERNS:
            topic = pattern.search(query)
            if topic and topic.group(0) not in evidence:
                return False
        return True

    @staticmethod
    def _confidence(sources: list[dict]) -> float:
        if not sources:
            return 0.1
        top = float(sources[0].get("score", 0))
        return round(min(0.98, 0.55 + top / (top + 5)), 2)

    def profile(self, student_id: str) -> dict:
        result = self.profiles.summary(student_id, self.graph.nodes)
        persisted=self.storage.all_mastery(student_id)
        if persisted:
            result["mastery"]=[{"id":x["concept_id"],"name":self.graph.nodes.get(x["concept_id"],{}).get("name",x["concept_id"]),"attempts":x["attempts"],"mastery":round(x["mastery"]*100)} for x in persisted]
            result["overall"]=round(sum(x["mastery"] for x in result["mastery"])/len(result["mastery"]))
            result["recommended"]=min(result["mastery"],key=lambda x:x["mastery"])
        result["learning_path"] = self.graph.learning_path(result["recommended"]["id"])
        result["plan"] = self.learning.plan(student_id)
        result["wrong_count"] = len(self.storage.wrong_book(student_id))
        return result

    def teacher_overview(self) -> dict:
        overview = self.profiles.class_summary(self.graph.nodes)
        overview.update({"questions": len(self.answer_logs), "low_confidence": sum(x["confidence"] < .6 for x in self.answer_logs), "feedback_count": len(self.feedback), "recent_questions": self.answer_logs[-8:][::-1]})
        return overview

    def add_feedback(self, payload: dict) -> None:
        self.feedback.append(payload)
        self.storage.add_feedback(payload)
        if payload.get("rating") == "needs_improvement":
            self.storage.quality_issue(payload.get("query",""), payload.get("answer",""), "teacher_feedback")

    def knowledge_entries(self) -> list[dict]:
        return json.loads((self.root / "knowledge_base" / "curriculum.json").read_text(encoding="utf-8"))

    def add_knowledge_entry(self, entry: dict) -> None:
        path=self.root/"knowledge_base"/"curriculum.json";entries=self.knowledge_entries()
        if any(x["id"]==entry["id"] for x in entries): raise ValueError("知识点 ID 已存在")
        entries.append(entry);path.write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding="utf-8")
        self.storage.quality_issue(entry["id"],entry["definition"],"knowledge_added")
        self.rag=LocalRAG(self.root/"knowledge_base"/"documents")
