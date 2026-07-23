import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ruyitutor.service import TutorService

ROOT = Path(__file__).resolve().parents[1]


class QuestionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.service = TutorService(ROOT)

    def test_dify_sources_are_used_before_local_rag(self):
        self.service.dify.dataset_key = "dataset-key"
        self.service.dify.dataset_id = "dataset-id"
        sources = [{
            "title": "代数式资料",
            "score": 0.8,
            "excerpt": "代数式是由数和表示数的字母通过运算符号连接而成的式子。",
        }]
        with patch.object(type(self.service.llm), "enabled", new_callable=PropertyMock, return_value=True), patch.object(
            self.service.dify,
            "retrieve",
            return_value=sources,
        ) as retrieve, patch.object(
            self.service.rag,
            "search",
            return_value=[],
        ) as local_search, patch.object(
            self.service.llm,
            "answer_from_sources",
            return_value="Dify 增强回答",
        ) as answer_from_sources:
            result = self.service.answer({
                "query": "给我讲讲代数式",
                "student_id": "t",
                "grade": "七年级",
                "subject": "数学",
                "mode": "启发辅导",
            })

        retrieve.assert_called_once()
        local_search.assert_not_called()
        answer_from_sources.assert_called_once()
        self.assertEqual(result["stage"], "dify-grounded")
        self.assertEqual(result["engine"], "Dify 知识库检索 + 大模型")

    def test_dify_failure_falls_back_to_local_rag(self):
        self.service.dify.dataset_key = "dataset-key"
        self.service.dify.dataset_id = "dataset-id"
        hit = self.service.rag.search("给我讲讲代数式", top_k=1)[0]
        with patch.object(type(self.service.llm), "enabled", new_callable=PropertyMock, return_value=True), patch.object(
            self.service.dify,
            "retrieve",
            side_effect=RuntimeError("dify unavailable"),
        ), patch.object(
            self.service.rag,
            "search",
            return_value=[hit],
        ), patch.object(
            self.service.llm,
            "answer",
            return_value="本地 RAG 增强回答",
        ):
            result = self.service.answer({
                "query": "给我讲讲代数式",
                "student_id": "t",
                "grade": "七年级",
                "subject": "数学",
                "mode": "启发辅导",
            })

        self.assertEqual(result["stage"], "local-grounded")
        self.assertEqual(result["engine"], "本地 RAG 备用 + 大模型")

    def test_common_difference_question_uses_local_rag_when_dify_misses(self):
        self.service.dify.dataset_key = "dataset-key"
        self.service.dify.dataset_id = "dataset-id"
        with patch.object(type(self.service.llm), "enabled", new_callable=PropertyMock, return_value=True), patch.object(
            self.service.dify,
            "retrieve",
            return_value=[],
        ), patch.object(
            self.service.llm,
            "answer",
            return_value="本地 RAG 增强回答",
        ):
            result = self.service.answer({
                "query": "速度和平均速度有什么区别",
                "student_id": "t",
                "grade": "八年级",
                "subject": "物理",
                "mode": "快速讲解",
            })

        self.assertEqual(result["stage"], "local-grounded")
        self.assertEqual(result["engine"], "本地 RAG 备用 + 大模型")
        self.assertTrue(result["sources"])
        self.assertIn("速度与平均速度", result["sources"][0]["title"])

    def test_missing_dify_and_local_knowledge_uses_general_llm(self):
        self.service.dify.dataset_key = "dataset-key"
        self.service.dify.dataset_id = "dataset-id"
        with patch.object(type(self.service.llm), "enabled", new_callable=PropertyMock, return_value=True), patch.object(
            self.service.dify,
            "retrieve",
            return_value=[],
        ), patch.object(
            self.service.rag,
            "search",
            return_value=[],
        ), patch.object(
            self.service.llm,
            "general_answer",
            return_value="通用大模型回答",
        ) as general_answer:
            result = self.service.answer({
                "query": "请解释量子纠缠的贝尔不等式实验和诺贝尔奖背景",
                "student_id": "t",
                "grade": "九年级",
                "subject": "物理",
                "mode": "快速讲解",
            })

        general_answer.assert_called_once()
        self.assertEqual(result["stage"], "llm-general")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], "通用大模型回答")


if __name__ == "__main__":
    unittest.main()
