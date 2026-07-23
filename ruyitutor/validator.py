from __future__ import annotations
import re
from .rag import tokenize

class AnswerValidator:
    UNIT_PATTERN = re.compile(r"\b(?:m/s|km/h|kg/m³|g/cm³|N|Pa|J|W|cm|km|kg|秒|分钟|小时)\b", re.I)

    def validate(self, answer: str, sources: list[dict]) -> dict:
        warnings = []
        if not sources:
            return {"grounded": False, "coverage": 0.0, "math_checked": False, "unit_checked": False, "warnings": ["没有可用检索证据"]}
        evidence = " ".join(str(s.get("excerpt", "")) for s in sources)
        answer_tokens, evidence_tokens = set(tokenize(answer)), set(tokenize(evidence))
        meaningful = {t for t in answer_tokens if len(t) >= 2}
        coverage = len(meaningful & evidence_tokens) / max(len(meaningful), 1)
        grounded = coverage >= 0.12
        if not grounded:
            warnings.append("回答与检索证据的词汇覆盖偏低，建议教师复核")
        equations = re.findall(r"(?<!\w)(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)", answer)
        math_ok = True
        ops = {"+":lambda a,b:a+b,"-":lambda a,b:a-b,"*":lambda a,b:a*b,"/":lambda a,b:a/b if b else float("inf")}
        for left, op, right, result in equations:
            if abs(ops[op](float(left),float(right))-float(result)) > 1e-6:
                math_ok = False; warnings.append(f"算式 {left}{op}{right}={result} 需要复核")
        units = self.UNIT_PATTERN.findall(answer)
        unit_ok = not units or all(unit in answer for unit in units)
        claims=[]
        for sentence in [x.strip() for x in re.split(r"[。！？\n]+",answer) if len(x.strip())>=8][:12]:
            tokens={t for t in tokenize(sentence) if len(t)>=2}
            support=len(tokens&evidence_tokens)/max(len(tokens),1)
            claims.append({"claim":sentence[:100],"support":round(support,3),"status":"supported" if support>=.1 else "review"})
        try:
            import sympy as sp
            for expression in re.findall(r"([0-9xX+\-*/^() ]{3,}=[0-9xX+\-*/^() ]+)",answer):
                left,right=expression.split("=",1);x=sp.symbols("x")
                sp.sympify(left.replace("^","**"),locals={"x":x});sp.sympify(right.replace("^","**"),locals={"x":x})
        except ImportError:
            pass
        except Exception:
            math_ok=False;warnings.append("检测到无法解析的数学表达式")
        return {"grounded": grounded, "coverage": round(coverage,3), "math_checked": math_ok, "unit_checked": unit_ok, "warnings": warnings,"claims":claims}
