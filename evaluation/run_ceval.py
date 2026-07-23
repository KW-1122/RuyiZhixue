from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_API_BASE_URL"])
model = os.environ["LLM_MODEL"]


def ask(item: dict) -> dict:
    prompt = "请只输出正确选项的单个大写字母A、B、C或D，不要解释。\n" + item["question"] + "\n" + "\n".join(f"{x}. {item[x]}" for x in "ABCD")
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=256)
    text = response.choices[0].message.content.strip().upper()
    matches = re.findall(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", text)
    predicted = matches[-1] if matches else ""
    return {"id": int(item["id"]), "predicted": predicted, "answer": item["answer"], "correct": predicted == item["answer"]}


def main() -> None:
    report = {"model": model, "dataset": "C-Eval CC BY-NC-SA 4.0", "splits": {}}
    for subject in ("middle_school_mathematics", "middle_school_physics"):
        rows = pd.read_parquet(ROOT / f"evaluation/public/ceval-exam/{subject}/val.parquet").to_dict("records")
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(ask, row) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda x: x["id"])
        report["splits"][subject] = {"samples": len(results), "accuracy": round(sum(x["correct"] for x in results) / len(results), 4), "details": results}
    total = sum(x["samples"] for x in report["splits"].values())
    correct = sum(sum(y["correct"] for y in x["details"]) for x in report["splits"].values())
    report["overall"] = {"samples": total, "accuracy": round(correct / total, 4)}
    output = ROOT / "evaluation/ceval_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "splits": {k: {"samples": v["samples"], "accuracy": v["accuracy"]} for k, v in report["splits"].items()}, "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
