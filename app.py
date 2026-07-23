from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import json
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ruyitutor.service import TutorService

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
service = TutorService(ROOT)

app = FastAPI(title="如意智学 RuyiTutor", version="1.0.0")
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    student_id: str = "demo-student"
    subject: str = "自动识别"
    grade: str = "八年级"
    mode: str = "启发辅导"
    conversation_id: str = ""
    reveal_answer: bool = False


class PracticeRequest(BaseModel):
    student_id: str = "demo-student"
    concept_id: str
    correct: bool

class ImageQuestionRequest(BaseModel):
    image_base64: str = ""
    correction: str = ""

class FeedbackRequest(BaseModel):
    student_id: str = "demo-student"
    query: str = ""
    rating: str
    note: str = ""

class FavoriteRequest(BaseModel):
    student_id: str = "demo-student"
    title: str
    content: str
    source: str = ""

class LoginRequest(BaseModel):
    user_id: str
    password: str

class PracticeSubmitRequest(BaseModel):
    student_id: str = "demo-student"
    question_id: str
    answer: str
    hints: int = 0

class ResolveIssueRequest(BaseModel):
    resolution: str

class KnowledgeEntryRequest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]{3,40}$")
    name: str = Field(min_length=2,max_length=50)
    subject: str
    grade: str
    source: str
    definition: str = Field(min_length=10)
    example: str = Field(min_length=5)
    mistake: str = Field(min_length=5)
    practice: str = Field(min_length=5)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": service.engine_name, "documents": len(service.rag.documents), "dify": service.dify.status()}


@app.get("/api/integrations/dify/status")
def dify_status(live: bool = False) -> dict:
    return service.dify.status(live=live)


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return service.answer(payload.model_dump())

@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    result=service.answer(payload.model_dump())
    def generate():
        yield json.dumps({"event":"meta","data":{k:v for k,v in result.items() if k!="answer"}},ensure_ascii=False)+"\n"
        for start in range(0,len(result["answer"]),28):
            yield json.dumps({"event":"text","data":result["answer"][start:start+28]},ensure_ascii=False)+"\n"
        yield json.dumps({"event":"done"},ensure_ascii=False)+"\n"
    return StreamingResponse(generate(),media_type="application/x-ndjson")


@app.get("/api/graph")
def graph() -> dict:
    return service.graph.public_view()


@app.get("/api/profile/{student_id}")
def profile(student_id: str) -> dict:
    return service.profile(student_id)


@app.post("/api/practice")
def practice(payload: PracticeRequest) -> dict:
    service.profiles.record(payload.student_id, payload.concept_id, payload.correct)
    return service.profile(payload.student_id)

@app.post("/api/ocr")
def ocr(payload: ImageQuestionRequest) -> dict:
    if len(payload.image_base64) > 14_000_000:
        raise HTTPException(413,"图片不能超过约 10MB")
    return service.ocr.extract(payload.image_base64, payload.correction)

@app.get("/api/teacher/overview")
def teacher_overview() -> dict:
    return service.teacher_overview()

@app.post("/api/teacher/feedback")
def teacher_feedback(payload: FeedbackRequest) -> dict:
    service.add_feedback(payload.model_dump())
    return {"status": "recorded"}

@app.get("/api/users")
def users() -> list[dict]:
    return service.storage.users()

@app.get("/api/history/{student_id}")
def history(student_id: str) -> list[dict]:
    return service.storage.history(student_id)

@app.get("/api/conversations/{student_id}")
def conversations(student_id: str) -> list[dict]:
    return service.storage.conversations(student_id)

@app.delete("/api/conversations/{student_id}/{conversation_id}")
def delete_conversation(student_id: str,conversation_id: str) -> dict:
    service.storage.delete_conversation(student_id,conversation_id);return {"status":"deleted"}

@app.post("/api/favorites")
def add_favorite(payload: FavoriteRequest) -> dict:
    service.storage.add_favorite(payload.student_id, payload.title, payload.content, payload.source)
    return {"status": "saved"}

@app.get("/api/favorites/{student_id}")
def favorites(student_id: str) -> list[dict]:
    return service.storage.favorites(student_id)

@app.post("/api/auth/login")
def login(payload: LoginRequest) -> dict:
    result=service.auth.login(payload.user_id,payload.password)
    return result or {"error":"用户名或密码错误"}

@app.get("/api/practice/next/{student_id}")
def next_practice(student_id: str, concept_id: str | None = None) -> dict:
    return service.learning.next_question(student_id,concept_id)

@app.post("/api/practice/submit")
def submit_practice(payload: PracticeSubmitRequest) -> dict:
    return service.learning.submit(payload.student_id,payload.question_id,payload.answer,payload.hints)

@app.get("/api/wrong-book/{student_id}")
def wrong_book(student_id: str) -> list[dict]:
    return service.storage.wrong_book(student_id)

@app.get("/api/learning-plan/{student_id}")
def learning_plan(student_id: str) -> dict:
    return service.learning.plan(student_id)

@app.get("/api/teacher/quality-issues")
def quality_issues() -> list[dict]:
    return service.storage.quality_issues()

@app.get("/api/teacher/student/{student_id}")
def teacher_student(student_id: str) -> dict:
    return {"profile":service.profile(student_id),"history":service.storage.history(student_id,20),"wrong_book":service.storage.wrong_book(student_id)}

@app.post("/api/teacher/quality-issues/{issue_id}/resolve")
def resolve_issue(issue_id: int,payload: ResolveIssueRequest) -> dict:
    service.storage.resolve_issue(issue_id,payload.resolution);return {"status":"resolved"}

@app.get("/api/teacher/knowledge")
def list_knowledge() -> list[dict]:
    return service.knowledge_entries()

@app.post("/api/teacher/knowledge")
def add_knowledge(payload: KnowledgeEntryRequest) -> dict:
    try: service.add_knowledge_entry(payload.model_dump())
    except ValueError as exc: raise HTTPException(409,str(exc)) from exc
    return {"status":"indexed","chunks":len(service.rag.documents)}


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
