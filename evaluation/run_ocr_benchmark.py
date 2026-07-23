from __future__ import annotations
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ruyitutor.ocr import OCRService
manifest=json.loads(Path(__file__).with_name("ocr_manifest.json").read_text(encoding="utf-8"));service=OCRService()
results=[]
for case in manifest:
    path=Path(__file__).parent/case["file"]
    if not path.exists():results.append({**case,"status":"sample_missing"});continue
    payload=base64.b64encode(path.read_bytes()).decode();result=service.extract(payload)
    results.append({**case,"status":"pass" if case["expected"] in result["text"] else "fail","result":result})
print(json.dumps(results,ensure_ascii=False,indent=2))
