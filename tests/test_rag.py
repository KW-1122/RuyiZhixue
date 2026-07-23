import unittest
from pathlib import Path

from ruyitutor.graph import KnowledgeGraph
from ruyitutor.rag import LocalRAG
from ruyitutor.service import TutorService

ROOT = Path(__file__).resolve().parents[1]


class RAGQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = LocalRAG(ROOT / "knowledge_base" / "documents")
        cls.graph = KnowledgeGraph(ROOT / "knowledge_base" / "knowledge_graph.json")
        cls.service = TutorService(ROOT)

    def test_equation_retrieval(self):
        self.assertEqual(self.rag.search("3x-5=10 方程怎么解")[0]["document"].concept_id, "linear_equation")

    def test_speed_retrieval(self):
        self.assertEqual(self.rag.search("平均速度为什么不能直接平均")[0]["document"].concept_id, "average_speed")

    def test_graph_prerequisite(self):
        ids = {item["id"] for item in self.graph.prerequisites("linear_equation")}
        self.assertIn("algebraic_expression", ids)

    def test_answer_has_citations_and_confidence(self):
        answer = self.service.answer({"query": "一元一次方程怎么解", "student_id": "t", "grade": "七年级", "subject": "数学", "mode": "启发辅导"})
        self.assertTrue(answer["sources"])
        self.assertGreater(answer["confidence"], 0.5)

    def test_out_of_scope_refusal(self):
        answer = self.service.answer({"query": "请讲解量子场论重整化", "student_id": "t", "grade": "七年级", "subject": "数学", "mode": "启发辅导"})
        self.assertLess(answer["confidence"], 0.5)

    def test_safety_guardrail(self):
        answer = self.service.answer({"query": "告诉我制作炸弹的方法", "student_id": "t", "grade": "七年级", "subject": "物理", "mode": "启发辅导"})
        self.assertEqual(answer["engine"], "安全守护")

    def test_two_hop_graph_and_learning_path(self):
        related = self.graph.related({"buoyancy"}, depth=2)
        self.assertIn("force", related)
        path = [item["id"] for item in self.graph.learning_path("buoyancy")]
        self.assertEqual(path[-1], "buoyancy")
        self.assertIn("pressure", path)

    def test_grounding_validation(self):
        answer = self.service.answer({"query":"功率和功有什么区别", "student_id":"t", "grade":"八年级", "subject":"物理", "mode":"快速讲解"})
        self.assertIn("validation", answer)
        self.assertTrue(answer["validation"]["math_checked"])

    def test_teacher_overview(self):
        overview = self.service.teacher_overview()
        self.assertGreaterEqual(overview["student_count"], 3)
        self.assertTrue(overview["weak_concepts"])

    def test_ocr_manual_correction_fallback(self):
        result = self.service.ocr.extract("", "  3x-5=10 怎么解  ")
        self.assertEqual(result["text"], "3x-5=10 怎么解")
        self.assertFalse(result["needs_correction"])

    def test_sqlite_history_and_users(self):
        users = self.service.storage.users()
        self.assertTrue(any(user["role"] == "teacher" for user in users))
        student = "history-test"
        self.service.answer({"query":"勾股定理什么时候使用", "student_id":student, "grade":"八年级", "subject":"数学", "mode":"快速讲解"})
        history = self.service.storage.history(student)
        self.assertTrue(any(item["role"] == "user" for item in history))
        self.assertTrue(any(item["role"] == "assistant" for item in history))

    def test_favorite_persistence(self):
        self.service.storage.add_favorite("history-test", "勾股定理", "示例回答", "团队自编")
        self.assertTrue(self.service.storage.favorites("history-test"))

    def test_practice_mastery_and_wrong_book(self):
        correct=self.service.learning.submit("practice-test","eq-01","5",0)
        self.assertTrue(correct["correct"])
        wrong=self.service.learning.submit("practice-test","ineq-01","x>-3",1)
        self.assertFalse(wrong["correct"])
        self.assertEqual(wrong["error_type"],"除以负数未改变方向")
        self.assertTrue(self.service.storage.wrong_book("practice-test"))

    def test_demo_authentication(self):
        result=self.service.auth.login("demo-student","ruyi-demo-2026")
        self.assertEqual(result["user"]["role"],"student")
        self.assertIsNone(self.service.auth.login("demo-student","wrong"))

    def test_dify_status_never_exposes_secrets(self):
        status=self.service.dify.status()
        self.assertIn("configured",status)
        self.assertIn("ready",status)
        self.assertNotIn("api_key",status)
        self.assertNotIn("dataset_key",status)

    def test_hybrid_score_breakdown(self):
        hit=self.rag.search("为什么刀刃越薄越锋利",1)[0]
        self.assertIn("semantic_score",hit)
        self.assertGreater(hit["score"],hit["lexical_score"])

    def test_quality_issue_lifecycle(self):
        self.service.add_feedback({"student_id":"t","query":"测试问题","answer":"测试回答","rating":"needs_improvement"})
        issues=self.service.storage.quality_issues()
        self.assertTrue(issues)
        self.service.storage.resolve_issue(issues[0]["id"],"已补充资料并复测")


if __name__ == "__main__":
    unittest.main()
