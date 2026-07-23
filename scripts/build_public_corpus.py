from __future__ import annotations

import json
import re
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("数学", ROOT / "knowledge_base/public_sources/义务教育数学课程标准2022.pdf", "中华人民共和国教育部《义务教育数学课程标准（2022年版）》"),
    ("物理", ROOT / "knowledge_base/public_sources/义务教育物理课程标准2022.pdf", "中华人民共和国教育部《义务教育物理课程标准（2022年版）》"),
]


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunks(text: str, size: int = 520, overlap: int = 80):
    for start in range(0, len(text), size - overlap):
        value = text[start:start + size].strip()
        if len(value) >= 120:
            yield value


records = []
for subject, path, citation in SOURCES:
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, 1):
        page_text = clean(page.extract_text() or "")
        for index, content in enumerate(chunks(page_text), 1):
            records.append({
                "id": f"moe-{subject}-{page_number}-{index}", "title": f"{subject}课程标准·第{page_number}页",
                "subject": subject, "grade": "义务教育", "concept_id": f"curriculum-standard-{subject}",
                "source": f"{citation}，第{page_number}页", "content": content,
            })

output = ROOT / "knowledge_base/public_curriculum_chunks.json"
if not records:
    raise SystemExit("课标PDF未包含可抽取文本；请先执行OCR，不生成空知识库。")
output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"chunks": len(records), "output": str(output)}, ensure_ascii=False))
