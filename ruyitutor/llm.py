from __future__ import annotations

import os
from openai import OpenAI


class GroundedLLM:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None and os.getenv("USE_LLM", "false").lower() == "true"

    def answer(self, query: str, evidence: list[dict], mode: str, prereqs: str) -> str:
        context = "\n\n".join(
            f"[{i}] {x['document'].title}\n{x['document'].content}"
            for i, x in enumerate(evidence, 1)
        )
        prompt = f"""你是面向中学生的教学辅导助手。只能依据下列检索证据回答，不得补造事实。
模式：{mode}；前置知识：{prereqs}
要求：先定位知识点，再分步骤解释；每个关键结论用[1]这类编号引用；证据不足时明确拒答；最后给一道自检题。

检索证据：
{context}

学生问题：{query}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "你是严谨、循序渐进且不会脱离证据的中小学教师。"}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()
