from __future__ import annotations
import json, sqlite3, hashlib
from datetime import datetime, timezone
from pathlib import Path

class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()

class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._init()

    def connect(self):
        con = sqlite3.connect(self.path, timeout=10, check_same_thread=False, factory=ClosingConnection)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        with self.connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,name TEXT NOT NULL,role TEXT NOT NULL DEFAULT 'student',grade TEXT);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,conversation_id TEXT,student_id TEXT,role TEXT,content TEXT,metadata TEXT,created_at TEXT);
            CREATE TABLE IF NOT EXISTS favorites(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,title TEXT,content TEXT,source TEXT,created_at TEXT);
            CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,query TEXT,rating TEXT,note TEXT,created_at TEXT);
            CREATE TABLE IF NOT EXISTS practice_records(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,concept_id TEXT,question_id TEXT,answer TEXT,correct INTEGER,difficulty REAL,hints INTEGER,error_type TEXT,created_at TEXT);
            CREATE TABLE IF NOT EXISTS concept_mastery(student_id TEXT,concept_id TEXT,mastery REAL,attempts INTEGER,last_practiced TEXT,PRIMARY KEY(student_id,concept_id));
            CREATE TABLE IF NOT EXISTS wrong_book(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id TEXT,question_id TEXT,concept_id TEXT,question TEXT,student_answer TEXT,correct_answer TEXT,error_type TEXT,resolved INTEGER DEFAULT 0,created_at TEXT);
            CREATE TABLE IF NOT EXISTS quality_issues(id INTEGER PRIMARY KEY AUTOINCREMENT,query TEXT,answer TEXT,issue_type TEXT,status TEXT DEFAULT 'open',resolution TEXT,created_at TEXT,updated_at TEXT);
            CREATE TABLE IF NOT EXISTS knowledge_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,document_id TEXT,action TEXT,content_hash TEXT,created_at TEXT);
            """)
            columns={r[1] for r in con.execute("PRAGMA table_info(users)")}
            if "password_hash" not in columns: con.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
            con.executemany("INSERT OR IGNORE INTO users(id,name,role,grade) VALUES(?,?,?,?)",[
                ("demo-student","小如同学","student","八年级"),("student-02","小明","student","八年级"),("student-03","小禾","student","七年级"),("teacher-01","李老师","teacher","")])
            default_hash=hashlib.sha256("ruyi-demo-2026".encode()).hexdigest()
            con.execute("UPDATE users SET password_hash=? WHERE password_hash IS NULL",(default_hash,))
            con.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(1,?)",(datetime.now(timezone.utc).isoformat(),))

    def users(self):
        with self.connect() as con: return [dict(r) for r in con.execute("SELECT * FROM users ORDER BY role,name")]

    def save_message(self, conversation_id, student_id, role, content, metadata=None):
        with self.connect() as con: con.execute("INSERT INTO messages(conversation_id,student_id,role,content,metadata,created_at) VALUES(?,?,?,?,?,?)",(conversation_id,student_id,role,content,json.dumps(metadata or {},ensure_ascii=False),datetime.now(timezone.utc).isoformat()))

    def history(self, student_id, limit=50):
        with self.connect() as con:
            rows=con.execute("SELECT * FROM messages WHERE student_id=? ORDER BY id DESC LIMIT ?",(student_id,limit)).fetchall()
        return [{**dict(r),"metadata":json.loads(r["metadata"] or "{}")} for r in rows]

    def conversations(self, student_id):
        with self.connect() as con:
            rows=con.execute("SELECT conversation_id,MIN(created_at) created_at,MAX(CASE WHEN role='user' THEN substr(content,1,40) END) title,COUNT(*) message_count FROM messages WHERE student_id=? GROUP BY conversation_id ORDER BY MIN(id) DESC",(student_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, student_id, conversation_id):
        with self.connect() as con: con.execute("DELETE FROM messages WHERE student_id=? AND conversation_id=?",(student_id,conversation_id))

    def add_favorite(self, student_id, title, content, source=""):
        with self.connect() as con: con.execute("INSERT INTO favorites(student_id,title,content,source,created_at) VALUES(?,?,?,?,?)",(student_id,title,content,source,datetime.now(timezone.utc).isoformat()))

    def favorites(self, student_id):
        with self.connect() as con: return [dict(r) for r in con.execute("SELECT * FROM favorites WHERE student_id=? ORDER BY id DESC",(student_id,))]

    def add_feedback(self, payload):
        with self.connect() as con: con.execute("INSERT INTO feedback(student_id,query,rating,note,created_at) VALUES(?,?,?,?,?)",(payload.get("student_id",""),payload.get("query",""),payload.get("rating",""),payload.get("note",""),datetime.now(timezone.utc).isoformat()))

    def authenticate(self, user_id: str, password: str):
        digest=hashlib.sha256(password.encode()).hexdigest()
        with self.connect() as con: row=con.execute("SELECT id,name,role,grade FROM users WHERE id=? AND password_hash=?",(user_id,digest)).fetchone()
        return dict(row) if row else None

    def record_practice(self, student_id, item, answer, correct, hints, error_type, mastery):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as con:
            con.execute("INSERT INTO practice_records(student_id,concept_id,question_id,answer,correct,difficulty,hints,error_type,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(student_id,item["concept_id"],item["id"],answer,int(correct),item["difficulty"],hints,error_type,now))
            old=con.execute("SELECT attempts FROM concept_mastery WHERE student_id=? AND concept_id=?",(student_id,item["concept_id"])).fetchone()
            attempts=(old[0] if old else 0)+1
            con.execute("INSERT INTO concept_mastery(student_id,concept_id,mastery,attempts,last_practiced) VALUES(?,?,?,?,?) ON CONFLICT(student_id,concept_id) DO UPDATE SET mastery=excluded.mastery,attempts=excluded.attempts,last_practiced=excluded.last_practiced",(student_id,item["concept_id"],mastery,attempts,now))
            if not correct:
                con.execute("INSERT INTO wrong_book(student_id,question_id,concept_id,question,student_answer,correct_answer,error_type,created_at) VALUES(?,?,?,?,?,?,?,?)",(student_id,item["id"],item["concept_id"],item["question"],answer,item["answer"],error_type,now))

    def mastery(self, student_id, concept_id):
        with self.connect() as con: row=con.execute("SELECT mastery,attempts,last_practiced FROM concept_mastery WHERE student_id=? AND concept_id=?",(student_id,concept_id)).fetchone()
        if not row:return {"mastery":0.35,"attempts":0,"last_practiced":None}
        result=dict(row)
        if result["last_practiced"]:
            last=datetime.fromisoformat(result["last_practiced"]);days=max(0,(datetime.now(timezone.utc)-last).days)
            result["mastery"]=max(.05,result["mastery"]*(.992**days))
        return result

    def all_mastery(self, student_id):
        with self.connect() as con: return [dict(r) for r in con.execute("SELECT * FROM concept_mastery WHERE student_id=? ORDER BY mastery",(student_id,))]

    def wrong_book(self, student_id):
        with self.connect() as con: return [dict(r) for r in con.execute("SELECT * FROM wrong_book WHERE student_id=? ORDER BY id DESC",(student_id,))]

    def quality_issue(self, query, answer, issue_type):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as con: con.execute("INSERT INTO quality_issues(query,answer,issue_type,created_at,updated_at) VALUES(?,?,?,?,?)",(query,answer,issue_type,now,now))

    def quality_issues(self):
        with self.connect() as con: return [dict(r) for r in con.execute("SELECT * FROM quality_issues ORDER BY status,id DESC")]

    def resolve_issue(self, issue_id, resolution):
        with self.connect() as con: con.execute("UPDATE quality_issues SET status='resolved',resolution=?,updated_at=? WHERE id=?",(resolution,datetime.now(timezone.utc).isoformat(),issue_id))

    def backup(self, target: Path):
        target.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as source, sqlite3.connect(target) as destination: source.backup(destination)
