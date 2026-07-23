from __future__ import annotations

import json
from pathlib import Path


class KnowledgeGraph:
    def __init__(self, path: Path):
        self.data = json.loads(path.read_text(encoding="utf-8"))
        curriculum_paths = [path.with_name("curriculum.json"), path.parent / "expanded" / "expanded_catalog.json"]
        for curriculum_path in curriculum_paths:
          if curriculum_path.exists():
            curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
            existing = {node["id"] for node in self.data["nodes"]}
            for item in curriculum:
                if item["id"] not in existing:
                    self.data["nodes"].append({"id": item["id"], "name": item["name"], "subject": item["subject"], "level": 2})
            prerequisite_map = {
                "integer_power":"negative_number", "algebra_simplify":"algebraic_expression",
                "inequality":"linear_equation", "proportion":"coordinate", "factorization":"algebra_simplify",
                "fraction_equation":"linear_equation", "quadratic_root":"integer_power",
                "similar_triangle":"triangle", "quadratic_function":"coordinate", "probability":"statistics",
                "pressure":"force", "buoyancy":"pressure", "work_power":"force", "density":"distance_time"
            }
            known = {(e["source"], e["target"]) for e in self.data["edges"]}
            for target, source in prerequisite_map.items():
                if (source, target) not in known:
                    self.data["edges"].append({"source": source, "target": target, "relation": "前置"})
        self.nodes = {node["id"]: node for node in self.data["nodes"]}

    def neighbors(self, concept_ids: set[str]) -> set[str]:
        found = set(concept_ids)
        for edge in self.data["edges"]:
            if edge["source"] in concept_ids or edge["target"] in concept_ids:
                found.update((edge["source"], edge["target"]))
        return found

    def related(self, concept_ids: set[str], depth: int = 2) -> set[str]:
        found, frontier = set(concept_ids), set(concept_ids)
        for _ in range(depth):
            nxt = set()
            for edge in self.data["edges"]:
                if edge["source"] in frontier:
                    nxt.add(edge["target"])
                if edge["target"] in frontier:
                    nxt.add(edge["source"])
            frontier = nxt - found
            found.update(frontier)
        return found

    def learning_path(self, concept_id: str) -> list[dict]:
        ordered, visiting = [], set()
        def visit(node_id: str) -> None:
            if node_id in visiting:
                return
            visiting.add(node_id)
            for item in self.prerequisites(node_id):
                visit(item["id"])
            if node_id in self.nodes:
                ordered.append(self.nodes[node_id])
        visit(concept_id)
        return ordered

    def prerequisites(self, concept_id: str) -> list[dict]:
        ids = [e["source"] for e in self.data["edges"] if e["target"] == concept_id and e["relation"] == "前置"]
        return [self.nodes[item] for item in ids if item in self.nodes]

    def public_view(self) -> dict:
        return self.data
