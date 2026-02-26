import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import nltk
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

from modules.answers import generate_suggestions
from modules.agent import run_agent_loop
from modules.config import (
    ANSWER_MODEL,
    CHAT_MODEL,
    CLASSIFIER_MODEL,
    EXTRACTION_MODEL,
    MAX_FILE_SIZE_MB,
    PERSONA_PATH,
)
from modules.document_processing import check_and_process_document
from modules.file_utils import check_file_size
from modules.memory import (
    init_short_term_memory,
    init_user_facts_memory,
    update_short_term_memory,
    update_user_facts_from_explicit_message,
)
from modules.persona import build_system_prompt, load_persona
from modules.text_processing import build_vector_db
from modules.tools import dispatch_tool_call, get_tool_map
from modules.utils import get_document_display_name


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


try:
    PERSONA = load_persona(PERSONA_PATH)
    PERSONA_ERROR = None
except (FileNotFoundError, ValueError) as exc:
    PERSONA = None
    PERSONA_ERROR = str(exc)

try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


@dataclass
class AppSession:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    processed: bool = False
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    conversation_context: List[Dict[str, Any]] = field(default_factory=list)
    short_term_memory: Dict[str, Any] = field(default_factory=init_short_term_memory)
    user_facts_memory: Dict[str, Any] = field(default_factory=init_user_facts_memory)
    suggested_questions: List[str] = field(default_factory=list)
    vector_db: Any = None
    chunks_metadata: List[Dict[str, Any]] = field(default_factory=list)
    embedding_model: Any = None
    injection_alerts: List[Dict[str, Any]] = field(default_factory=list)


SESSIONS: Dict[str, AppSession] = {}
SESSIONS_LOCK = Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_ui_message(role: str, content: str, *, debug: Optional[Dict[str, Any]] = None,
                    suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": _utc_now_iso(),
    }
    if suggestions:
        msg["suggestions"] = suggestions[:3]
    if debug:
        msg["debug"] = debug
    return msg


def _session_phase(session: AppSession) -> str:
    if not session.processed:
        return "upload"
    if session.chat_history:
        return "chat"
    return "dashboard"


def _persona_to_ui(persona: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not persona:
        return {
            "name": "Candidate",
            "role": "Persona unavailable",
            "location": "",
            "tone": "",
            "bio": "",
            "avatarUrl": "",
            "contactLinks": [],
        }
    contact = persona.get("contact", {}) or {}
    links = []
    for label, key in (
        ("GitHub", "github"),
        ("LinkedIn", "linkedin"),
        ("Website", "website"),
        ("Email", "email"),
    ):
        value = contact.get(key)
        if not value:
            continue
        url = value if key != "email" or value.startswith("mailto:") else f"mailto:{value}"
        links.append({"label": label, "url": url})
    return {
        "name": persona.get("name", "Candidate"),
        "role": persona.get("role", "Candidate"),
        "location": persona.get("location", ""),
        "tone": persona.get("tone", ""),
        "bio": persona.get("bio", ""),
        "avatarUrl": "",
        "contactLinks": links,
    }


def _doc_to_ui(doc: Dict[str, Any]) -> Dict[str, Any]:
    highlights: List[str] = []
    for entry in (doc.get("work_history") or [])[:3]:
        title = entry.get("title") or "Role"
        org = entry.get("organization") or "Organization"
        start = entry.get("start_display") or "?"
        end = entry.get("end_display") or "?"
        highlights.append(f"{title} at {org} ({start}-{end})")
    if not highlights:
        for item in (doc.get("extracted_data", {}).get("key_facts") or [])[:3]:
            if item:
                highlights.append(str(item))

    detected_type = (
        doc.get("detected_type")
        or doc.get("classification", {}).get("type")
        or doc.get("file_type", "document")
    )
    return {
        "id": doc.get("doc_id") or str(doc.get("index") or uuid.uuid4()),
        "name": doc.get("name") or get_document_display_name(doc),
        "fileType": doc.get("file_type", "non_resume"),
        "detectedType": detected_type,
        "riskScore": float(doc.get("injection_report", {}).get("risk_score", 0) or 0),
        "confidence": float(doc.get("classification", {}).get("confidence", 0) or 0),
        "metadataSummary": str(
            doc.get("metadata", {}).get("summary")
            or doc.get("classification", {}).get("justification")
            or ""
        ),
        "extractedHighlights": highlights or None,
    }


def _alert_to_ui(alert: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": alert.get("id") or str(uuid.uuid4()),
        "documentName": alert.get("documentName", "Document"),
        "riskScore": float(alert.get("riskScore", 0) or 0),
        "message": alert.get("message", ""),
        "timestamp": alert.get("timestamp") or _utc_now_iso(),
    }


def _debug_to_ui(debug_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not debug_info:
        return None

    errors: List[str] = []
    if debug_info.get("error"):
        errors.append(str(debug_info["error"]))

    tool_calls = []
    for call in debug_info.get("tool_calls", []) or []:
        args = call.get("args")
        if isinstance(args, dict):
            args_text = json.dumps(args, ensure_ascii=False, default=str)
        else:
            args_text = str(args or "")
        if len(args_text) > 110:
            args_text = args_text[:107] + "..."
        tool_calls.append(
            {
                "tool": str(call.get("tool", "")),
                "args": args_text,
                "resultCount": int(call.get("result_count", 0) or 0),
                "error": call.get("error"),
            }
        )

    return {
        "agentRounds": int(debug_info.get("agent_rounds", 0) or 0),
        "toolCalls": tool_calls,
        "finishReason": str(debug_info.get("finish_reason") or ("error" if errors else "stop")),
        "errors": errors or None,
    }


def _state_to_ui(session_id: str, session: AppSession) -> Dict[str, Any]:
    return {
        "sessionId": session_id,
        "phase": _session_phase(session),
        "processed": session.processed,
        "documents": [_doc_to_ui(doc) for doc in session.documents],
        "alerts": [_alert_to_ui(alert) for alert in session.injection_alerts],
        "chatHistory": session.chat_history,
        "suggestedQuestions": session.suggested_questions,
    }


def _config_to_ui() -> Dict[str, Any]:
    return {
        "persona": _persona_to_ui(PERSONA),
        "personaError": PERSONA_ERROR,
        "apiKeyPresent": bool(OPENAI_API_KEY),
        "modelConfig": {
            "classifier": CLASSIFIER_MODEL,
            "extraction": EXTRACTION_MODEL,
            "answer": ANSWER_MODEL,
            "chat": CHAT_MODEL,
        },
        "availableTools": [
            {"name": tool_name.replace("_", " ").title(), "status": "active"}
            for tool_name in get_tool_map().keys()
        ],
        "maxFileSizeMB": MAX_FILE_SIZE_MB,
    }


def _get_or_create_session(session_id: Optional[str]) -> Tuple[str, AppSession]:
    with SESSIONS_LOCK:
        if session_id and session_id in SESSIONS:
            return session_id, SESSIONS[session_id]
        new_id = str(uuid.uuid4())
        SESSIONS[new_id] = AppSession()
        return new_id, SESSIONS[new_id]


def _require_session(session_id: str) -> AppSession:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _require_client() -> OpenAI:
    if not OPENAI_CLIENT:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured on the backend")
    return OPENAI_CLIENT


def _make_system_prompt(session: AppSession) -> str:
    if not PERSONA:
        return "You are a helpful assistant."

    doc_parts = []
    for d in session.documents:
        name = get_document_display_name(d)
        doc_type = d.get("detected_type", d.get("file_type", "document"))
        summary = d.get("metadata", {}).get("summary", "")
        line = f"[{doc_type}] {name}: {summary}"

        work_history = d.get("work_history", [])
        if work_history:
            wh_lines = []
            for entry in work_history[:8]:
                title = entry.get("title", "Role")
                org = entry.get("organization", "?")
                start = entry.get("start_display", "?")
                end = entry.get("end_display", "?")
                wh_lines.append(f"  • {title} at {org} ({start} - {end})")
            line += "\n  Work/Education:\n" + "\n".join(wh_lines)

        doc_parts.append(line)

    return build_system_prompt(PERSONA, "\n".join(doc_parts))


def _call_tool_for_session(session: AppSession, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return dispatch_tool_call(
        tool_name,
        arguments,
        vector_db=session.vector_db,
        chunks_metadata=session.chunks_metadata,
        embedding_model=session.embedding_model,
        persona=PERSONA,
    )


def _handle_chat(session: AppSession, question: str, client: OpenAI) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    answer, debug_info = run_agent_loop(
        question=question,
        system_prompt=_make_system_prompt(session),
        client=client,
        call_tool=lambda tool_name, arguments: _call_tool_for_session(session, tool_name, arguments),
        short_term_memory=session.short_term_memory,
        user_facts_memory=session.user_facts_memory,
    )

    session.short_term_memory = update_short_term_memory(
        session.short_term_memory,
        question,
        answer or "",
    )
    session.user_facts_memory = update_user_facts_from_explicit_message(
        session.user_facts_memory,
        question,
    )

    session.conversation_context.append(
        {
            "question": question,
            "answer_snippet": (answer or "")[:200],
        }
    )
    if len(session.conversation_context) > 5:
        session.conversation_context.pop(0)

    persona_name = PERSONA["name"] if PERSONA else "the candidate"
    session.suggested_questions = generate_suggestions(
        question,
        answer,
        persona_name,
        session.documents,
        client,
    )

    user_msg = _new_ui_message("user", question)
    assistant_msg = _new_ui_message(
        "assistant",
        answer or "",
        debug=_debug_to_ui(debug_info),
        suggestions=session.suggested_questions,
    )
    session.chat_history.extend([user_msg, assistant_msg])
    return assistant_msg, debug_info


def _build_alert(filename: str, risk_score: float) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "documentName": filename,
        "riskScore": float(risk_score),
        "message": f"Suspicious content detected (risk: {risk_score:.1%})",
        "timestamp": _utc_now_iso(),
    }


def _read_uploads_to_memory(files: List[UploadFile]) -> List[NamedBytesIO]:
    named_files: List[NamedBytesIO] = []
    for upload in files:
        if not upload.filename:
            continue
        data = upload.file.read()
        named_files.append(NamedBytesIO(data, upload.filename))
    return named_files


def _process_documents(session: AppSession, files: List[NamedBytesIO], *, append: bool, client: OpenAI) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    valid_files: List[NamedBytesIO] = []
    validation_errors: List[str] = []
    for file_obj in files:
        if not file_obj.name.lower().endswith(".pdf"):
            validation_errors.append(f"{file_obj.name}: Only PDF files are supported")
            continue
        is_valid, _, err = check_file_size(file_obj)
        if not is_valid:
            validation_errors.append(f"{file_obj.name}: {err}")
            continue
        valid_files.append(file_obj)

    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="No valid PDF files to process",
        )

    working_docs = list(session.documents) if append else []
    new_alerts: List[Dict[str, Any]] = []
    warnings: List[str] = []
    added_count = 0
    start_index = len(working_docs)

    for i, file_obj in enumerate(valid_files):
        file_obj.seek(0)
        doc_data, is_dup, dup_info, inj_report, _classification = check_and_process_document(
            file_obj,
            start_index + i,
            working_docs,
            client,
        )

        if is_dup:
            warnings.append(f"Skipped {file_obj.name}: {dup_info['message']}")
            continue

        working_docs.append(doc_data)
        added_count += 1

        if inj_report and inj_report.risk_score > 0.3:
            new_alerts.append(_build_alert(file_obj.name, float(inj_report.risk_score)))

    if added_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents were added (duplicates or invalid files)",
        )

    vector_db, chunks_meta, model = build_vector_db(working_docs)
    session.documents = working_docs
    session.vector_db = vector_db
    session.chunks_metadata = chunks_meta
    session.embedding_model = model
    session.processed = True
    session.user_facts_memory = init_user_facts_memory(PERSONA, session.documents)
    session.suggested_questions = []

    if append:
        session.injection_alerts.extend(new_alerts)
    else:
        session.injection_alerts = new_alerts
        session.chat_history = []
        session.conversation_context = []
        session.short_term_memory = init_short_term_memory()

    return {
        "addedCount": added_count,
        "warnings": warnings,
        "validationErrors": validation_errors,
    }


class ChatRequest(BaseModel):
    sessionId: str
    question: str


class SessionRequest(BaseModel):
    sessionId: str


app = FastAPI(title="PersonaChat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": _utc_now_iso(),
        "personaLoaded": bool(PERSONA),
        "apiKeyPresent": bool(OPENAI_API_KEY),
    }


@app.get("/api/bootstrap")
def bootstrap(session_id: Optional[str] = None) -> Dict[str, Any]:
    sid, session = _get_or_create_session(session_id)
    return {
        "config": _config_to_ui(),
        "state": _state_to_ui(sid, session),
    }


@app.post("/api/reset")
def reset_session(payload: SessionRequest) -> Dict[str, Any]:
    with SESSIONS_LOCK:
        if payload.sessionId not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        SESSIONS[payload.sessionId] = AppSession()
        session = SESSIONS[payload.sessionId]
    return {"state": _state_to_ui(payload.sessionId, session)}


@app.post("/api/process")
def process_documents(
    sessionId: str = Form(...),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    session = _require_session(sessionId)
    client = _require_client()
    named_files = _read_uploads_to_memory(files)
    result = _process_documents(session, named_files, append=False, client=client)
    return {"state": _state_to_ui(sessionId, session), "result": result}


@app.post("/api/add-documents")
def add_documents(
    sessionId: str = Form(...),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    session = _require_session(sessionId)
    client = _require_client()
    if not session.processed:
        raise HTTPException(status_code=400, detail="Process initial documents first")
    named_files = _read_uploads_to_memory(files)
    result = _process_documents(session, named_files, append=True, client=client)
    return {"state": _state_to_ui(sessionId, session), "result": result}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    session = _require_session(payload.sessionId)
    client = _require_client()
    if not session.processed or not session.documents:
        raise HTTPException(status_code=400, detail="No processed documents available")
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    assistant_msg, _debug = _handle_chat(session, question, client)
    return {
        "assistantMessage": assistant_msg,
        "state": _state_to_ui(payload.sessionId, session),
    }
