from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from ruyitutor.dify import DifyClient
from ruyitutor.graph import KnowledgeGraph
from ruyitutor.llm import GroundedLLM
from ruyitutor.rag import LocalRAG, tokenize

CASES = json.loads((ROOT / "evaluation" / "eval_cases.json").read_text(encoding="utf-8"))
DOMAIN_CASES = [case for case in CASES if case.get("expected")]
GENERATION_INDICES = [0, 2, 5, 10, 16, 18, 19, 22]


def ranked_ids(hits: list[dict]) -> list[str]:
    return [hit["document"].concept_id for hit in hits]


def retrieval_metrics(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error")]
    ranks = [int(row["rank"]) for row in valid]
    return {
        "cases": len(rows),
        "successful": len(valid),
        "availability": round(len(valid) / len(rows), 4),
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / max(len(ranks), 1), 4),
        "recall_at_3": round(sum(0 < rank <= 3 for rank in ranks) / max(len(ranks), 1), 4),
        "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / max(len(ranks), 1), 4),
        "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in valid) / max(len(valid), 1), 2),
    }


def run_retrieval(rag: LocalRAG, graph: KnowledgeGraph, dify: DifyClient) -> tuple[dict, list[dict]]:
    groups: dict[str, list[dict]] = {name: [] for name in (
        "keyword_rag", "hybrid_rag", "hybrid_graphrag", "dify_hybrid_rerank"
    )}
    original_scores = rag.vector_store.scores
    for case in DOMAIN_CASES:
        query, expected = case["query"], case["expected"]

        rag.vector_store.scores = lambda _query: {}
        started = time.perf_counter()
        hits = rag.search(query, top_k=3)
        groups["keyword_rag"].append(make_retrieval_row(case, expected, hits, started))
        rag.vector_store.scores = original_scores

        started = time.perf_counter()
        hits = rag.search(query, top_k=3)
        groups["hybrid_rag"].append(make_retrieval_row(case, expected, hits, started))

        started = time.perf_counter()
        seed = rag.search(query, top_k=3)
        related = graph.related(set(ranked_ids(seed)), depth=2)
        hits = rag.search(query, top_k=3, concept_boost=related)
        groups["hybrid_graphrag"].append(make_retrieval_row(case, expected, hits, started))

        started = time.perf_counter()
        try:
            remote = dify.retrieve(query, top_k=3)
            ids = [infer_concept_id(item, rag) for item in remote]
            rank = ids.index(expected) + 1 if expected in ids else 0
            row = {**case, "top3": ids, "rank": rank,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 2), "error": ""}
        except Exception as exc:
            row = {**case, "top3": [], "rank": 0,
                   "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                   "error": type(exc).__name__}
        groups["dify_hybrid_rerank"].append(row)
    rag.vector_store.scores = original_scores
    return {name: retrieval_metrics(rows) for name, rows in groups.items()}, [
        {"group": name, **row} for name, rows in groups.items() for row in rows
    ]


def make_retrieval_row(case: dict, expected: str, hits: list[dict], started: float) -> dict:
    ids = ranked_ids(hits)
    rank = ids.index(expected) + 1 if expected in ids else 0
    return {**case, "top3": ids, "rank": rank,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "error": ""}


def infer_concept_id(item: dict, rag: LocalRAG) -> str:
    text = " ".join(str(item.get(key, "")) for key in ("title", "content", "excerpt", "source"))
    for doc in rag.documents:
        if doc.id in text or doc.concept_id in text or doc.title in text:
            return doc.concept_id
    return "unknown"


def evidence_text(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {hit['document'].title}\n{hit['document'].content}" for i, hit in enumerate(hits, 1))


def call_llm(llm: GroundedLLM, query: str, hits: list[dict] | None, graph_context: str = "") -> str:
    if not llm.client:
        raise RuntimeError("LLM_API_KEY is not configured")
    if hits is None:
        prompt = f"请直接回答这道初中数学或物理问题：{query}。给出知识定位、分步讲解和一道自检题。"
    else:
        prompt = f"""你是初中辅导老师。只能依据证据回答，不得补造；关键结论用[1]格式引用。
图谱提供的前置关系：{graph_context or '未使用'}
证据：
{evidence_text(hits)}

问题：{query}
请给出知识定位、分步讲解和一道自检题。"""
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def score_answer(answer: str, reference: str, has_evidence: bool, latency_ms: float) -> dict:
    answer_tokens, reference_tokens = Counter(tokenize(answer)), Counter(tokenize(reference))
    important = [token for token, count in reference_tokens.most_common(30) if len(token) >= 2]
    coverage = sum(token in answer_tokens for token in important) / max(len(important), 1)
    structure_terms = ("知识", "步骤", "自检", "试一试", "定位", "讲解")
    structure = sum(term in answer for term in structure_terms) / len(structure_terms)
    citation = bool(re.search(r"\[\d+\]", answer)) if has_evidence else False
    return {
        "reference_coverage": round(coverage, 4),
        "pedagogical_structure": round(structure, 4),
        "citation_present": citation,
        "answer_length": len(answer),
        "latency_ms": round(latency_ms, 2),
    }


def run_generation(rag: LocalRAG, graph: KnowledgeGraph, dify: DifyClient, llm: GroundedLLM) -> tuple[dict, list[dict]]:
    selected = [DOMAIN_CASES[index] for index in GENERATION_INDICES]
    rows: list[dict] = []
    for case in selected:
        query, expected = case["query"], case["expected"]
        reference_docs = [doc for doc in rag.documents if doc.concept_id == expected]
        reference = "\n".join(doc.content for doc in reference_docs)
        normal_hits = rag.search(query, top_k=4)
        related = graph.related(set(ranked_ids(normal_hits[:3])), depth=2)
        graph_hits = rag.search(query, top_k=4, concept_boost=related)
        prereqs = "、".join(node["name"] for node in graph.prerequisites(expected))

        jobs = (
            ("pure_llm", lambda: call_llm(llm, query, None), False),
            ("hybrid_rag_llm", lambda: call_llm(llm, query, normal_hits), True),
            ("hybrid_graphrag_llm", lambda: call_llm(llm, query, graph_hits, prereqs), True),
            ("full_dify_rag", lambda: dify.chat(query, {"grade": "八年级", "subject": "自动识别", "mode": "快速讲解"})["answer"], True),
        )
        for group, job, has_evidence in jobs:
            started = time.perf_counter()
            try:
                answer = job()
                latency = (time.perf_counter() - started) * 1000
                score = score_answer(answer, reference, has_evidence, latency)
                rows.append({"group": group, **case, "answer": answer, "error": "", **score})
            except Exception as exc:
                rows.append({"group": group, **case, "answer": "", "error": type(exc).__name__,
                             "reference_coverage": 0, "pedagogical_structure": 0,
                             "citation_present": False, "answer_length": 0,
                             "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
    summary = {}
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        valid = [row for row in subset if not row["error"]]
        summary[group] = {
            "cases": len(subset), "successful": len(valid),
            "reference_coverage": avg(valid, "reference_coverage"),
            "pedagogical_structure": avg(valid, "pedagogical_structure"),
            "citation_rate": round(sum(row["citation_present"] for row in valid) / max(len(valid), 1), 4),
            "avg_latency_ms": avg(valid, "latency_ms"),
        }
    return summary, rows


def avg(rows: list[dict], key: str) -> float:
    return round(sum(float(row[key]) for row in rows) / max(len(rows), 1), 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_report(report: dict) -> None:
    lines = ["# 如意智学 RAG 消融实验报告", "", f"运行时间：{report['run_at']}", "",
             "## 1. 实验设计", "",
             "检索层使用全部 23 道领域题；回答层使用 8 道代表题。所有组共享题目、知识库和评分程序，每次只改变检索/图谱组件。", "",
             "## 2. 检索层结果", "",
             "| 实验组 | 成功/总数 | 可用率 | Hit@1 | Recall@3 | MRR | 平均耗时(ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["retrieval_summary"].items():
        lines.append(f"| {name} | {item['successful']}/{item['cases']} | {item['availability']:.2%} | {item['hit_at_1']:.2%} | {item['recall_at_3']:.2%} | {item['mrr']:.4f} | {item['avg_latency_ms']:.2f} |")
    lines += ["", "## 3. 回答层结果", "",
              "| 实验组 | 成功/总数 | 参考要点覆盖 | 教学结构 | 引用率 | 平均耗时(ms) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["generation_summary"].items():
        lines.append(f"| {name} | {item['successful']}/{item['cases']} | {item['reference_coverage']:.2%} | {item['pedagogical_structure']:.2%} | {item['citation_rate']:.2%} | {item['avg_latency_ms']:.2f} |")
    lines += ["", "## 4. 评分口径与限制", "",
              "- Hit@1、Recall@3、MRR 根据预先标注的目标知识点计算。",
              "- 参考要点覆盖率是答案与目标知识文档关键 token 的重合率，不等同于人工正确率。",
              "- 教学结构检查知识定位、步骤讲解和自检提示等显式结构。",
              "- 引用率只检查具有检索证据的组是否输出编号引用。",
              "- 原始答案完整保存在 `ablation_generation.csv`，答辩结论应结合人工抽检。", "",
              "## 5. 可复现命令", "", "```powershell", "python evaluation/run_ablation.py", "```", ""]
    (ROOT / "evaluation" / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_chart(report: dict) -> None:
    labels = {"pure_llm": "Pure LLM", "hybrid_rag_llm": "Hybrid RAG",
              "hybrid_graphrag_llm": "GraphRAG", "full_dify_rag": "Dify RAG"}
    items = [(labels.get(name, name), values["reference_coverage"])
             for name, values in report["generation_summary"].items()]
    width, height, left, top, chart_h = 900, 470, 180, 70, 310
    bar_w = 115
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f7f5ee"/>',
             '<text x="40" y="35" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#173f35">RuyiTutor RAG Ablation: Reference Coverage</text>']
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        parts += [f'<line x1="{left}" y1="{y}" x2="850" y2="{y}" stroke="#d7dfda"/>',
                  f'<text x="135" y="{y+5}" font-family="Arial" font-size="13" fill="#65736e">{tick}%</text>']
    colors = ["#9aa7a2", "#5b9b83", "#1f6b55", "#e98b54"]
    for index, (label, value) in enumerate(items):
        x = left + 55 + index * 155
        bar_h = chart_h * value
        y = top + chart_h - bar_h
        parts += [f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="8" fill="{colors[index % len(colors)]}"/>',
                  f'<text x="{x+bar_w/2}" y="{y-10}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#173f35">{value:.1%}</text>',
                  f'<text x="{x+bar_w/2}" y="{top+chart_h+30}" text-anchor="middle" font-family="Arial" font-size="14" fill="#263b35">{label}</text>']
    parts += ['<text x="450" y="450" text-anchor="middle" font-family="Arial" font-size="12" fill="#71807c">8 representative questions; automatic reference-token coverage, with raw outputs retained for manual review.</text>', '</svg>']
    (ROOT / "evaluation" / "ablation_chart.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    if "--summarize-existing" in sys.argv:
        output = ROOT / "evaluation"
        retrieval_rows = list(csv.DictReader((output / "ablation_retrieval.csv").open(encoding="utf-8-sig")))
        generation_rows = list(csv.DictReader((output / "ablation_generation.csv").open(encoding="utf-8-sig")))
        retrieval_summary = {group: retrieval_metrics([row for row in retrieval_rows if row["group"] == group])
                             for group in sorted({row["group"] for row in retrieval_rows})}
        generation_summary = {}
        for group in sorted({row["group"] for row in generation_rows}):
            subset = [row for row in generation_rows if row["group"] == group]
            valid = [row for row in subset if not row["error"]]
            generation_summary[group] = {
                "cases": len(subset), "successful": len(valid),
                "reference_coverage": avg(valid, "reference_coverage"),
                "pedagogical_structure": avg(valid, "pedagogical_structure"),
                "citation_rate": round(sum(row["citation_present"].lower() == "true" for row in valid) / max(len(valid), 1), 4),
                "avg_latency_ms": avg(valid, "latency_ms"),
            }
        report = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "knowledge_chunks": 79,
                  "retrieval_summary": retrieval_summary, "generation_summary": generation_summary,
                  "notes": {"generation_cases": len(GENERATION_INDICES),
                            "automatic_scores_require_manual_review": True,
                            "failed_cloud_calls_excluded_from_quality_metrics": True}}
        (output / "ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(report)
        write_chart(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    rag = LocalRAG(ROOT / "knowledge_base" / "documents")
    graph = KnowledgeGraph(ROOT / "knowledge_base" / "knowledge_graph.json")
    dify, llm = DifyClient(), GroundedLLM()
    retrieval_summary, retrieval_rows = run_retrieval(rag, graph, dify)
    generation_summary, generation_rows = run_generation(rag, graph, dify, llm)
    report = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "knowledge_chunks": len(rag.documents),
              "retrieval_summary": retrieval_summary,
              "generation_summary": generation_summary,
              "notes": {"generation_cases": len(GENERATION_INDICES), "automatic_scores_require_manual_review": True}}
    output = ROOT / "evaluation"
    (output / "ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "ablation_retrieval.csv", retrieval_rows)
    write_csv(output / "ablation_generation.csv", generation_rows)
    write_report(report)
    write_chart(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
