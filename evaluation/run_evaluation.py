from __future__ import annotations
import json, sys
from dotenv import load_dotenv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from ruyitutor.rag import LocalRAG
from ruyitutor.service import TutorService
CASES = json.loads(Path(__file__).with_name("eval_cases.json").read_text(encoding="utf-8"))

def main() -> None:
    rag, service = LocalRAG(ROOT / "knowledge_base" / "documents"), TutorService(ROOT)
    retrieval_cases = [c for c in CASES if c["expected"]]
    recall_hits, reciprocal_ranks, details = 0, [], []
    for case in retrieval_cases:
        ids = [h["document"].concept_id for h in rag.search(case["query"], top_k=3)]
        rank = ids.index(case["expected"]) + 1 if case["expected"] in ids else 0
        recall_hits += bool(rank); reciprocal_ranks.append(1 / rank if rank else 0)
        details.append({"query":case["query"],"expected":case["expected"],"top3":ids,"rank":rank})
    base = {"student_id":"eval","grade":"八年级","subject":"自动识别","mode":"快速讲解"}
    out = next(c for c in CASES if c["type"] == "out_of_scope")
    danger = next(c for c in CASES if c["type"] == "safety")
    report = {"knowledge_chunks":len(rag.documents),"retrieval_cases":len(retrieval_cases),
              "recall_at_3":round(recall_hits/len(retrieval_cases),4),"mrr":round(sum(reciprocal_ranks)/len(reciprocal_ranks),4),
              "out_of_scope_refusal":service.answer({**base,"query":out["query"]})["stage"]=="refused",
              "safety_guardrail":service.answer({**base,"query":danger["query"]})["stage"]=="safety","details":details}
    Path(__file__).with_name("latest_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="details"},ensure_ascii=False,indent=2))
    if report["recall_at_3"] < .80 or not report["out_of_scope_refusal"] or not report["safety_guardrail"]: raise SystemExit("评测未达到质量门槛")
if __name__ == "__main__": main()
