from __future__ import annotations

from collections import defaultdict


class ProfileStore:
    def __init__(self):
        self._records: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    def record(self, student_id: str, concept_id: str, correct: bool) -> None:
        self._records[student_id][concept_id].append(correct)

    def summary(self, student_id: str, graph_nodes: dict[str, dict]) -> dict:
        records = self._records[student_id]
        mastery = []
        for concept_id, attempts in records.items():
            rate = sum(attempts) / len(attempts)
            mastery.append({
                "id": concept_id,
                "name": graph_nodes.get(concept_id, {}).get("name", concept_id),
                "attempts": len(attempts),
                "mastery": round((0.35 + 0.65 * rate) * 100),
            })
        if not mastery:
            mastery = [
                {"id": "linear_equation", "name": "一元一次方程", "attempts": 3, "mastery": 76},
                {"id": "negative_number", "name": "有理数运算", "attempts": 4, "mastery": 62},
                {"id": "average_speed", "name": "平均速度", "attempts": 2, "mastery": 48},
            ]
        overall = round(sum(item["mastery"] for item in mastery) / len(mastery))
        weakest = min(mastery, key=lambda item: item["mastery"])
        return {"overall": overall, "mastery": mastery, "recommended": weakest}

    def class_summary(self, graph_nodes: dict[str, dict]) -> dict:
        seed = {
            "demo-student": {"linear_equation":[1,1,0,1], "average_speed":[1,0]},
            "student-02": {"negative_number":[1,0,0], "pressure":[1,1,0]},
            "student-03": {"pythagorean":[1,1,1], "force":[0,0,1]},
        }
        for sid, concepts in seed.items():
            if sid not in self._records:
                for concept, attempts in concepts.items():
                    self._records[sid][concept].extend(bool(x) for x in attempts)
        student_ids = sorted(self._records)
        students = [{"student_id": sid, **self.summary(sid, graph_nodes)} for sid in student_ids]
        aggregate: dict[str, list[int]] = defaultdict(list)
        for student in students:
            for item in student["mastery"]:
                aggregate[item["id"]].append(item["mastery"])
        weak = sorted(({
            "id": cid, "name": graph_nodes.get(cid, {}).get("name", cid),
            "mastery": round(sum(scores)/len(scores)), "students": len(scores),
        } for cid, scores in aggregate.items()), key=lambda x: x["mastery"])
        return {"student_count": len(students), "class_average": round(sum(s["overall"] for s in students)/len(students)), "weak_concepts": weak[:5], "students": students}
