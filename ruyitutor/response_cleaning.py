from __future__ import annotations

import re


THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning tags that some models expose in their final answer."""
    cleaned = THINK_BLOCK.sub("", text or "")
    cleaned = THINK_TAG.sub("", cleaned)
    return cleaned.strip()
