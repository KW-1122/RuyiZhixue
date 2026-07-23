from __future__ import annotations

import os

from openai import OpenAI

from .response_cleaning import strip_thinking


class GroundedLLM:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None and os.getenv("USE_LLM", "false").lower() == "true"

    def _complete(self, messages: list[dict], max_tokens: int = 900) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return strip_thinking(response.choices[0].message.content)

    def answer(self, query: str, evidence: list[dict], mode: str, prereqs: str) -> str:
        sources = [{
            "title": item["document"].title,
            "excerpt": item["document"].content,
        } for item in evidence]
        return self.answer_from_sources(query, sources, mode, prereqs, "本地知识库")

    def answer_from_sources(
        self,
        query: str,
        sources: list[dict],
        mode: str,
        prereqs: str = "",
        source_name: str = "知识库",
    ) -> str:
        context = "\n\n".join(
            f"[{i}] {item.get('title', '资料')}\n{item.get('excerpt') or item.get('content') or ''}"
            for i, item in enumerate(sources, 1)
        )
        prompt = f"""你是面向中学生的教学辅导助手。请优先依据下列{source_name}检索资料回答，不要补造资料中没有的事实。
模式：{mode}
前置知识：{prereqs or "无"}

要求：
1. 先定位知识点，再分步骤解释。
2. 关键结论尽量用[1]这类编号引用资料。
3. 如果检索资料只覆盖部分问题，请明确说明哪些部分来自资料、哪些部分需要通用解释。
4. 最后给一道自检题。
5. 不要输出 <think> 内容。

检索资料：
{context}

学生问题：{query}"""
        return self._complete(
            [
                {"role": "system", "content": "你是严谨、循序渐进且尊重检索证据的中小学教师。"},
                {"role": "user", "content": prompt},
            ]
        )

    def general_answer(self, query: str, mode: str, grade: str, subject: str) -> str:
        prompt = f"""学生问题：{query}
年级：{grade}
学科：{subject}
模式：{mode}

知识库没有可靠命中。请直接用你的通用能力回答这个问题，使用适合中学生的表达。
如果题目较复杂，先给核心结论，再给步骤或例子。不要引用知识库编号，不要声称来自知识库，不要输出 <think> 内容。"""
        return self._complete(
            [
                {"role": "system", "content": "你是一个面向中学生的通用教学助手，回答要清晰、直接、不过度赘述。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )
