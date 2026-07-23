from __future__ import annotations
import json, math, re
from datetime import datetime, timezone
from pathlib import Path

def normalize(value: str) -> str:
    return re.sub(r"\s+","",value.lower()).replace("×","*").replace("²","2")

class LearningEngine:
    def __init__(self, root: Path, storage, graph):
        self.storage,self.graph=storage,graph
        self.items=json.loads((root/"knowledge_base"/"practice_bank.json").read_text(encoding="utf-8"))
        self.by_id={x["id"]:x for x in self.items}

    def next_question(self, student_id, concept_id=None):
        mastery={x["concept_id"]:x["mastery"] for x in self.storage.all_mastery(student_id)}
        candidates=[x for x in self.items if not concept_id or x["concept_id"]==concept_id] or self.items
        candidates.sort(key=lambda x:abs(x["difficulty"]-mastery.get(x["concept_id"],.35)))
        return {k:v for k,v in candidates[0].items() if k not in ("answer","error_rules")}

    def submit(self, student_id, question_id, answer, hints=0):
        item=self.by_id[question_id]; got,want=normalize(answer),normalize(item["answer"])
        correct=got==want or (want.split()[0] and got==want.split()[0] and not re.search(r"[a-z/³]",want))
        error_type="" if correct else next((label for pattern,label in item.get("error_rules",{}).items() if normalize(pattern)==got),"概念或计算错误")
        old=self.storage.mastery(student_id,item["concept_id"])
        prior=float(old["mastery"]); evidence=(1 if correct else 0)-min(hints,.3)*.15
        gain=(.22+.16*item["difficulty"])*(1-prior) if correct else -( .18+.1*item["difficulty"])*prior
        mastery=max(.05,min(.98,prior+gain-0.04*hints))
        self.storage.record_practice(student_id,item,answer,correct,hints,error_type,mastery)
        variant=self.next_question(student_id,item["concept_id"])
        return {"correct":correct,"error_type":error_type,"explanation":item["explanation"],"mastery":round(mastery*100),"variant":variant}

    def plan(self, student_id):
        rows=self.storage.all_mastery(student_id)
        if rows: target=min(rows,key=lambda x:x["mastery"])["concept_id"]
        else: target="linear_equation"
        path=self.graph.learning_path(target)
        return {"target":target,"path":path,"tasks":[{"type":"复习","concept":x["name"],"minutes":8} for x in path[-3:]]+[ {"type":"练习","concept":self.graph.nodes.get(target,{}).get("name",target),"minutes":12}]}
