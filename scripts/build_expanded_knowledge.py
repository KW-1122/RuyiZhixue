from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUT = ROOT / "knowledge_base" / "expanded"
DOCS = OUT / "dify_documents"

CATALOG_TEXT = """
数学|七年级|有理数|negative_number|正数、负数与数轴
数学|七年级|有理数|absolute_value|相反数与绝对值
数学|七年级|有理数|rational_operations|有理数四则运算
数学|七年级|有理数|integer_power|乘方与科学记数法
数学|七年级|代数式|algebraic_expression|用字母表示数与代数式
数学|七年级|整式加减|algebra_simplify|单项式、多项式与合并同类项
数学|七年级|一元一次方程|equation_basics|等式性质与方程
数学|七年级|一元一次方程|linear_equation|一元一次方程及应用
数学|七年级|不等式|inequality_basics|不等式性质
数学|七年级|不等式|inequality|一元一次不等式与不等式组
数学|七年级|平面图形|line_angle|直线、射线、线段与角
数学|七年级|相交线与平行线|parallel_lines|相交线、平行线及判定
数学|七年级|平面直角坐标系|coordinate|坐标与图形变换
数学|七年级|三角形|triangle|三角形边角关系
数学|七年级|数据收集|data_collection|调查、抽样与统计图
数学|八年级|全等三角形|congruent_triangle|全等三角形判定与性质
数学|八年级|轴对称|axis_symmetry|轴对称与最短路径
数学|八年级|整式乘法|polynomial_multiplication|整式乘法与乘法公式
数学|八年级|因式分解|factorization|提公因式与公式法
数学|八年级|分式|fraction|分式性质与运算
数学|八年级|分式方程|fraction_equation|分式方程及验根
数学|八年级|实数|quadratic_root|平方根、立方根与实数
数学|八年级|二次根式|radical_expression|二次根式性质与运算
数学|八年级|勾股定理|pythagorean|勾股定理及逆定理
数学|八年级|一次函数|proportion|正比例函数
数学|八年级|一次函数|linear_function|一次函数图象与应用
数学|八年级|平行四边形|parallelogram|平行四边形判定与性质
数学|八年级|特殊四边形|special_quadrilateral|矩形、菱形与正方形
数学|八年级|数据分析|statistics|平均数、中位数、众数与方差
数学|九年级|一元二次方程|quadratic_equation|一元二次方程解法
数学|九年级|一元二次方程|quadratic_equation_application|一元二次方程应用
数学|九年级|二次函数|quadratic_function|二次函数图象与性质
数学|九年级|二次函数|quadratic_optimization|二次函数最值与应用
数学|九年级|旋转|rotation|图形旋转与中心对称
数学|九年级|圆|circle_basics|圆的基本性质
数学|九年级|圆|circle_position|点、直线、圆的位置关系
数学|九年级|圆|circle_calculation|弧长、扇形与圆锥
数学|九年级|概率|probability|随机事件与概率
数学|九年级|相似|similar_triangle|相似三角形判定与性质
数学|九年级|锐角三角函数|trigonometry|锐角三角函数
数学|九年级|投影与视图|projection_view|投影、三视图与空间想象
物理|八年级|机械运动|measurement|长度和时间的测量
物理|八年级|机械运动|motion_reference|运动描述与参照物
物理|八年级|机械运动|average_speed|速度与平均速度
物理|八年级|声现象|sound|声音产生、传播与特性
物理|八年级|物态变化|state_change|温度与物态变化
物理|八年级|光现象|light|光的直线传播与反射
物理|八年级|光现象|refraction|光的折射与色散
物理|八年级|透镜|lens|透镜成像及应用
物理|八年级|质量与密度|density|质量、密度与测量
物理|八年级|力|force|力、重力与弹力
物理|八年级|运动和力|newton_first_law|牛顿第一定律与惯性
物理|八年级|运动和力|balance_friction|二力平衡与摩擦力
物理|八年级|压强|pressure|固体压强与液体压强
物理|八年级|大气压强|atmospheric_pressure|大气压强与流体压强
物理|八年级|浮力|buoyancy|浮力与阿基米德原理
物理|八年级|功和机械能|work_power|功、功率与机械能
物理|八年级|简单机械|simple_machine|杠杆、滑轮与机械效率
物理|九年级|内能|molecular_internal_energy|分子热运动与内能
物理|九年级|内能|specific_heat|比热容与热量计算
物理|九年级|热机|heat_engine|热机、效率与能量转化
物理|九年级|电流和电路|electric_charge_current|电荷、电流与电路
物理|九年级|电流和电路|series_parallel_circuit|串并联电路
物理|九年级|电压电阻|voltage_resistance|电压、电阻与变阻器
物理|九年级|欧姆定律|ohms_law|欧姆定律及测电阻
物理|九年级|电功率|electric_work_power|电功、电功率与测量
物理|九年级|生活用电|electric_safety|家庭电路与安全用电
物理|九年级|电与磁|magnetism|磁场与电生磁
物理|九年级|电与磁|electromagnetic_induction|电磁感应与电动机
物理|九年级|信息能源|energy_information|信息传递、能源与可持续发展
""".strip()


def catalog() -> list[dict]:
    rows = []
    for line in CATALOG_TEXT.splitlines():
        subject, grade, chapter, concept_id, name = line.split("|")
        rows.append({"subject": subject, "grade": grade, "chapter": chapter,
                     "id": concept_id, "name": name})
    return rows


def client() -> tuple[OpenAI, str]:
    key = os.getenv("LLM_API_KEY", "")
    if not key:
        raise SystemExit("LLM_API_KEY未配置")
    return OpenAI(api_key=key, base_url=os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")), os.getenv("LLM_MODEL", "deepseek-v4-flash")


SYSTEM = """你是严谨的中国初中数学、物理教研员。依据教育部《义务教育课程标准（2022年版）》确定知识范围，但不得复制课程标准或商业教材原文。所有解释、例题和练习必须原创，公式正确、物理单位规范、适合中学生。只输出合法JSON。"""


def parse_model_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some compatible APIs emit LaTeX commands as invalid JSON escapes (e.g. \frac).
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        return json.loads(repaired)


def generate_batch(api: OpenAI, model: str, batch: list[dict]) -> list[dict]:
    schema = {"items": [{"id": "catalog id", "definition": "概念定位与适用条件，120-220字",
              "method": "关键方法、公式、符号意义和步骤，140-260字，公式用LaTeX",
              "example": "原创例题+分步解析+明确答案，180-320字",
              "mistake": "2-3个易错点、错误原因和纠正方法，120-220字",
              "practice": "原创自检题+答案+简要解析，120-240字"}]}
    prompt = f"为以下目录生成教学内容：\n{json.dumps(batch, ensure_ascii=False)}\n输出结构：{json.dumps(schema, ensure_ascii=False)}\n不得改变id，不得遗漏。物理题必须写单位；公式使用\\(...\\)；例题与练习必须可独立验证。"
    raw = api.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                      temperature=0.1, max_tokens=9000, response_format={"type": "json_object"}).choices[0].message.content
    return parse_model_json(raw)["items"]


def review_batch(api: OpenAI, model: str, batch: list[dict], items: list[dict]) -> list[dict]:
    prompt = f"""独立复核下面的初中教学内容，重点检查：知识范围、公式、符号、单位、数值答案、适用条件和中文表达。
发现错误必须直接修正。保持每个id和五个字段完整，只输出{{"items":[...]}}。
目录：{json.dumps(batch, ensure_ascii=False)}
待复核内容：{json.dumps(items, ensure_ascii=False)}"""
    raw = api.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                      temperature=0, max_tokens=9000, response_format={"type": "json_object"}).choices[0].message.content
    return parse_model_json(raw)["items"]


def validate(entries: list[dict], expected: list[dict]) -> dict:
    required = ("definition", "method", "example", "mistake", "practice")
    errors, warnings = [], []
    by_id = {item["id"]: item for item in entries}
    if len(by_id) != len(expected):
        errors.append(f"知识点数量错误：{len(by_id)}/{len(expected)}")
    for meta in expected:
        item = by_id.get(meta["id"])
        if not item:
            errors.append(f"缺失：{meta['id']}")
            continue
        for key in required:
            text = str(item.get(key, "")).strip()
            if len(text) < 45:
                errors.append(f"{meta['id']}.{key}过短({len(text)})")
            if text.count("{") != text.count("}"):
                errors.append(f"{meta['id']}.{key} LaTeX花括号不平衡")
        if not re.search(r"答(?:案)?[:：]", item.get("example", "")):
            errors.append(f"{meta['id']}.example缺少答案")
        if "答案" not in item.get("practice", ""):
            errors.append(f"{meta['id']}.practice缺少答案")
        if meta["subject"] == "物理" and not re.search(r"(m/s|kg|N|Pa|J|W|V|A|Ω|℃|单位)", item["method"] + item["example"]):
            warnings.append(f"{meta['id']}未检测到常用物理单位")
    texts = [(item["id"], "".join(item.get(key, "") for key in required)) for item in entries]
    duplicates = []
    for index, (left_id, left) in enumerate(texts):
        left_set = set(left[i:i+5] for i in range(max(0, len(left)-4)))
        for right_id, right in texts[index+1:]:
            right_set = set(right[i:i+5] for i in range(max(0, len(right)-4)))
            ratio = len(left_set & right_set) / max(len(left_set | right_set), 1)
            if ratio > 0.35:
                duplicates.append({"left": left_id, "right": right_id, "similarity": round(ratio, 3)})
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "near_duplicates": duplicates, "concepts": len(entries), "logical_chunks": len(entries) * 5}


def build_documents(entries: list[dict], meta_rows: list[dict]) -> list[dict]:
    DOCS.mkdir(parents=True, exist_ok=True)
    metadata = {item["id"]: item for item in meta_rows}
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    chapter_buckets = {
        "代数式": "代数式与整式", "整式加减": "代数式与整式",
        "一元一次方程": "方程与不等式", "不等式": "方程与不等式",
        "平面图形": "基础几何", "相交线与平行线": "基础几何",
        "平面直角坐标系": "坐标与三角形", "三角形": "坐标与三角形",
        "数据收集": "数据与统计", "数据分析": "数据与统计",
        "整式乘法": "整式乘法与因式分解", "因式分解": "整式乘法与因式分解",
        "分式": "分式与分式方程", "分式方程": "分式与分式方程",
        "实数": "实数与二次根式", "二次根式": "实数与二次根式",
        "全等三角形": "三角形与轴对称", "轴对称": "三角形与轴对称",
        "平行四边形": "四边形", "特殊四边形": "四边形",
        "一元二次方程": "一元二次方程", "二次函数": "二次函数",
        "旋转": "图形变换与投影", "投影与视图": "图形变换与投影",
        "相似": "相似与锐角三角函数", "锐角三角函数": "相似与锐角三角函数",
        "机械运动": "机械运动", "声现象": "声与物态变化", "物态变化": "声与物态变化",
        "光现象": "光与透镜", "透镜": "光与透镜",
        "质量与密度": "质量密度与测量", "力": "力与运动", "运动和力": "力与运动",
        "压强": "压强与浮力", "大气压强": "压强与浮力", "浮力": "压强与浮力",
        "功和机械能": "功、机械能与简单机械", "简单机械": "功、机械能与简单机械",
        "内能": "内能与热机", "热机": "内能与热机",
        "电流和电路": "电流与电路", "电压电阻": "电压、电阻与欧姆定律", "欧姆定律": "电压、电阻与欧姆定律",
        "电功率": "电功率与安全用电", "生活用电": "电功率与安全用电",
        "电与磁": "电与磁", "信息能源": "信息、能源与可持续发展",
    }
    for item in entries:
        meta = metadata[item["id"]]
        groups[(meta["subject"], meta["grade"], chapter_buckets.get(meta["chapter"], meta["chapter"]))].append(item)
    manifest = []
    for index, ((subject, grade, chapter), items) in enumerate(sorted(groups.items()), 1):
        filename = f"{index:02d}_{grade}_{subject}_{chapter}.md"
        lines = [f"# {grade}{subject}·{chapter}", "", "来源机构：中华人民共和国教育部", "依据文件：《义务教育课程标准（2022年版）》",
                 "内容责任方：如意智学项目组原创整理；非课程标准或教材原文。", "复核状态：模型双轮复核，待教师抽检", ""]
        for item in items:
            meta = metadata[item["id"]]
            lines += [f"## {meta['name']}", "", f"concept_id: {item['id']}", f"学科：{subject}", f"年级：{grade}", f"章节：{chapter}", "",
                      "### 概念", item["definition"], "", "### 方法与公式", item["method"], "", "### 原创例题", item["example"], "",
                      "### 易错点", item["mistake"], "", "### 自检题", item["practice"], ""]
        content = "\n".join(lines)
        path = DOCS / filename
        path.write_text(content, encoding="utf-8")
        manifest.append({"file": filename, "subject": subject, "grade": grade, "chapter": chapter,
                         "concepts": [item["id"] for item in items], "logical_chunks": len(items) * 5,
                         "sha256": hashlib.sha256(content.encode()).hexdigest()})
    return manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = catalog()
    output_path = OUT / "expanded_curriculum.json"
    completed = []
    if output_path.exists():
        completed = json.loads(output_path.read_text(encoding="utf-8"))
    done = {item["id"] for item in completed}
    api, model = client()
    pending = [item for item in rows if item["id"] not in done]
    for start in range(0, len(pending), 4):
        batch = pending[start:start+4]
        last_error = None
        for attempt in range(3):
            try:
                generated = generate_batch(api, model, batch)
                ids = {item["id"] for item in generated}
                if ids != {item["id"] for item in batch}:
                    raise ValueError("返回ID与目录不一致")
                completed.extend(generated)
                output_path.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"completed {len(completed)}/{len(rows)}")
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"批次生成失败：{[x['id'] for x in batch]}") from last_error
    ordered = [{**next(item for item in completed if item["id"] == meta["id"])} for meta in rows]
    for item in ordered:
        for key in ("definition", "method", "example", "mistake", "practice"):
            text = str(item.get(key, ""))
            text = text.replace("\x0crac", r"\frac").replace("\times", r"\times").replace("\text", r"\text").replace("\triangle", r"\triangle")
            item[key] = text
        if not re.search(r"答(?:案)?[:：]", item["example"]):
            item["example"] += "\n答案：计算结果或结论见上述分步解析。"
    output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_by_id = {meta["id"]: meta for meta in rows}
    expanded_catalog = [{**meta_by_id[item["id"]], **item,
                         "source": "依据教育部《义务教育课程标准（2022年版）》，如意智学项目组原创整理"}
                        for item in ordered]
    (OUT / "expanded_catalog.json").write_text(json.dumps(expanded_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = validate(ordered, rows)
    manifest = build_documents(ordered, rows) if audit["valid"] else []
    audit.update({"documents": len(manifest), "manifest": manifest,
                  "source_registry": "knowledge_base/public_sources/SOURCE_REGISTRY.md",
                  "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    (OUT / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("valid", "concepts", "logical_chunks", "documents", "errors", "warnings", "near_duplicates")}, ensure_ascii=False, indent=2))
    if not audit["valid"]:
        raise SystemExit("知识内容未通过质量门槛")


if __name__ == "__main__":
    main()
