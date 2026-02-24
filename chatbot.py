import streamlit as st
import nltk
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from modules.config import (
    CHAT_MODEL, ANSWER_MODEL, EXTRACTION_MODEL, CLASSIFIER_MODEL,
    MAX_FILE_SIZE_MB, PERSONA_PATH,
)
from modules.models import FileType
from modules.utils import get_document_display_name
from modules.file_utils import check_file_size
from modules.session_state import get_default_state, perform_complete_reset
from modules.persona import load_persona, build_system_prompt
from modules.text_processing import build_vector_db
from modules.document_processing import check_and_process_document
from modules.answers import generate_suggestions
from modules.memory import (
    init_user_facts_memory,
    update_short_term_memory,
    update_user_facts_from_explicit_message,
)
from modules.tools import dispatch_tool_call, get_tool_map
from modules.agent import run_agent_loop


# =============================================================================
# LOAD PERSONA (single-tenant identity)
# =============================================================================

try:
    PERSONA = load_persona(PERSONA_PATH)
except (FileNotFoundError, ValueError) as exc:
    PERSONA = None
    _PERSONA_ERROR = str(exc)
else:
    _PERSONA_ERROR = None


# =============================================================================
# TOOL DISPATCH (session-state-aware wrapper)
# =============================================================================

def call_tool(tool_name: str, arguments: dict) -> dict:
    """Dispatch a tool call, automatically injecting session-state context.

    This is the single entry-point the agent loop (or any caller) should use
    to invoke a tool.  It pulls vector_db / chunks_metadata / embedding_model
    from Streamlit session state and persona from the module-level PERSONA.
    """
    return dispatch_tool_call(
        tool_name,
        arguments,
        vector_db=st.session_state.get("vector_db"),
        chunks_metadata=st.session_state.get("chunks_metadata"),
        embedding_model=st.session_state.get("embedding_model"),
        persona=PERSONA,
    )


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle_question(question, client, system_prompt):
    answer, debug_info = run_agent_loop(
        question=question,
        system_prompt=system_prompt,
        client=client,
        call_tool=call_tool,
        short_term_memory=st.session_state.short_term_memory,
        user_facts_memory=st.session_state.user_facts_memory,
    )

    st.session_state.short_term_memory = update_short_term_memory(
        st.session_state.short_term_memory,
        question,
        answer or "",
    )
    st.session_state.user_facts_memory = update_user_facts_from_explicit_message(
        st.session_state.user_facts_memory,
        question,
    )

    st.session_state.conversation_context.append({
        "question": question,
        "answer_snippet": answer[:200] if answer else "",
    })
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context.pop(0)

    persona_name = PERSONA["name"] if PERSONA else "the candidate"
    st.session_state.suggested_questions = generate_suggestions(
        question, answer, persona_name, st.session_state.documents, client,
    )

    return answer, debug_info


# =============================================================================
# APP SHUTDOWN
# =============================================================================

def shutdown_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Shutting down...")
    os._exit(0)

if st.session_state.get("shutdown_requested", False):
    shutdown_app()


# =============================================================================
# PAGE CONFIG & INITIALIZATION
# =============================================================================

persona_name = PERSONA["name"] if PERSONA else "Candidate"
st.set_page_config(
    page_title=f"Chat with {persona_name}",
    page_icon="💬",
    layout="wide",
)
st.title(f"💬 Chat with {persona_name}")
st.caption("Recruiter-facing agent — answers as the candidate using their own documents")

defaults = get_default_state()
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.documents and not st.session_state.user_facts_memory.get("key_facts"):
    st.session_state.user_facts_memory = init_user_facts_memory(
        PERSONA, st.session_state.documents
    )

try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


# =============================================================================
# SIDEBAR
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")

    # Persona status
    if PERSONA:
        st.success(f"🎭 Persona: **{PERSONA['name']}**")
        if PERSONA.get("role"):
            st.caption(f"Role: {PERSONA['role']}")
    else:
        st.error(f"🎭 Persona not loaded: {_PERSONA_ERROR}")

    # API key
    if OPENAI_API_KEY:
        st.success("✅ API Key (from .env)")
    else:
        st.warning("⚠️ Set OPENAI_API_KEY in your .env file")

    st.divider()
    st.subheader("🤖 Models")
    st.caption(f"Classifier: {CLASSIFIER_MODEL}")
    st.caption(f"Extraction: {EXTRACTION_MODEL}")
    st.caption(f"Answer: {ANSWER_MODEL}")

    st.divider()
    st.subheader("🛠️ Agent Tools")
    _tool_names = list(get_tool_map().keys())
    for _tn in _tool_names:
        _label = _tn.replace("_", " ").title()
        st.caption(f"• {_label}")
    st.caption(f"{len(_tool_names)} tool(s) available")

    if st.session_state.documents:
        st.divider()
        st.subheader(f"📚 {len(st.session_state.documents)} Document(s)")
        for d in st.session_state.documents:
            name = get_document_display_name(d)
            doc_type = d.get("detected_type", d.get("file_type", "unknown"))
            risk = d.get("injection_report", {}).get("risk_score", 0)
            risk_icon = "🔴" if risk > 0.5 else "🟡" if risk > 0.2 else "🟢"
            st.write(f"• {name} ({doc_type}) {risk_icon}")

    # Security alerts
    if st.session_state.get("injection_alerts"):
        st.divider()
        st.subheader("⚠️ Security Alerts")
        for alert in st.session_state.injection_alerts[-3:]:
            st.warning(alert, icon="🛡️")

    st.divider()
    confirm_exit = st.checkbox("Quit app", key="confirm_exit")
    if confirm_exit:
        if st.button("🛑 Quit", type="primary"):
            st.session_state.shutdown_requested = True
            st.rerun()

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# =============================================================================
# BUILD SYSTEM PROMPT (persona + loaded documents)
# =============================================================================

def _make_system_prompt():
    if not PERSONA:
        return "You are a helpful assistant."
    doc_parts = []
    for d in st.session_state.documents:
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
                wh_lines.append(f"  • {title} at {org} ({start} – {end})")
            line += "\n  Work/Education:\n" + "\n".join(wh_lines)

        doc_parts.append(line)
    return build_system_prompt(PERSONA, "\n".join(doc_parts))


# =============================================================================
# MAIN UI
# =============================================================================

if not st.session_state.processed:
    st.subheader("📤 Upload Your Documents")
    st.info(
        f"Upload your CV, LinkedIn export, cover letters, or any supporting documents. "
        f"Max {MAX_FILE_SIZE_MB} MB per file."
    )

    files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

    if files:
        valid_files = []
        for f in files:
            is_valid, size_mb, error = check_file_size(f)
            if is_valid:
                valid_files.append(f)
            else:
                st.error(f"❌ {f.name}: {error}")

        if valid_files:
            st.success(f"✅ {len(valid_files)} file(s) ready")

            if st.button("🚀 Process Documents", type="primary"):
                if not OPENAI_API_KEY:
                    st.error("Set OPENAI_API_KEY in your .env file")
                else:
                    progress = st.progress(0)
                    status = st.empty()
                    processed = []
                    injection_alerts = []

                    for i, f in enumerate(valid_files):
                        status.text(f"Processing: {f.name}")

                        doc_data, is_dup, dup_info, inj_report, classification = (
                            check_and_process_document(f, i, processed, client)
                        )

                        if is_dup:
                            st.warning(f"⚠️ Skipped {f.name}: {dup_info['message']}")
                        else:
                            if inj_report and inj_report.risk_score > 0.3:
                                alert_msg = (
                                    f"⚠️ {f.name}: Suspicious content detected "
                                    f"(risk: {inj_report.risk_score:.1%})"
                                )
                                st.warning(alert_msg)
                                injection_alerts.append(alert_msg)

                                if inj_report.suspicious_phrases:
                                    with st.expander(f"🔍 Details for {f.name}"):
                                        st.write("Suspicious patterns found:")
                                        for phrase in inj_report.suspicious_phrases[:5]:
                                            st.code(phrase)

                            if classification:
                                type_icon = (
                                    "📄" if classification.file_type == FileType.RESUME else "📁"
                                )
                                st.info(
                                    f"{type_icon} {f.name}: Classified as "
                                    f"**{classification.detected_document_type}** "
                                    f"({classification.confidence:.0%} confident)"
                                )

                            processed.append(doc_data)

                        progress.progress((i + 1) / len(valid_files))

                    status.empty()

                    if processed:
                        vector_db, chunks_meta, model = build_vector_db(processed)
                        st.session_state.documents = processed
                        st.session_state.vector_db = vector_db
                        st.session_state.chunks_metadata = chunks_meta
                        st.session_state.embedding_model = model
                        st.session_state.processed = True
                        st.session_state.injection_alerts = injection_alerts
                        st.session_state.user_facts_memory = init_user_facts_memory(
                            PERSONA, processed
                        )

                        st.success(f"✅ Processed {len(processed)} document(s)")
                        st.rerun()
                    else:
                        st.error("No documents processed.")


if st.session_state.processed:
    st.subheader(f"📚 {len(st.session_state.documents)} Document(s) Loaded")

    high_risk = [
        d for d in st.session_state.documents
        if d.get("injection_report", {}).get("risk_score", 0) > 0.3
    ]
    if high_risk:
        st.warning(f"⚠️ {len(high_risk)} document(s) have elevated injection risk scores")

    cols = st.columns(min(len(st.session_state.documents), 4))
    for i, d in enumerate(st.session_state.documents):
        with cols[i % 4]:
            with st.container(border=True):
                icon = "📄" if d.get("file_type") == "resume" else "📁"
                title = get_document_display_name(d)
                st.write(f"{icon} **{title}**")
                st.caption(
                    f"Type: {d.get('detected_type', d.get('file_type', 'unknown'))}"
                )
                risk = d.get("injection_report", {}).get("risk_score", 0)
                risk_color = "🔴" if risk > 0.5 else "🟡" if risk > 0.2 else "🟢"
                st.caption(f"Security: {risk_color} {risk:.0%}")

    col1, col2 = st.columns([1, 4])
    if col1.button("🗑️ Reset All", type="secondary"):
        perform_complete_reset()
        st.rerun()

    with st.expander("📤 Add More Documents"):
        new_files = st.file_uploader(
            "More PDFs", type=["pdf"], accept_multiple_files=True, key="add"
        )
        if new_files:
            valid = [f for f in new_files if check_file_size(f)[0]]
            if valid and st.button("➕ Add"):
                start = len(st.session_state.documents)
                added = 0
                for i, f in enumerate(valid):
                    doc_data, is_dup, dup_info, inj_report, classification = (
                        check_and_process_document(
                            f, start + i, st.session_state.documents, client
                        )
                    )
                    if is_dup:
                        st.warning(f"⚠️ Skipped {f.name}: {dup_info['message']}")
                    else:
                        if inj_report and inj_report.risk_score > 0.3:
                            st.warning(
                                f"⚠️ {f.name}: Suspicious content "
                                f"(risk: {inj_report.risk_score:.1%})"
                            )
                        st.session_state.documents.append(doc_data)
                        added += 1

                if added > 0:
                    vector_db, chunks_meta, model = build_vector_db(
                        st.session_state.documents
                    )
                    st.session_state.vector_db = vector_db
                    st.session_state.chunks_metadata = chunks_meta
                    st.session_state.embedding_model = model
                    st.session_state.user_facts_memory = init_user_facts_memory(
                        PERSONA, st.session_state.documents
                    )
                    st.success(f"Added {added} document(s)")
                st.rerun()

    st.divider()
    st.subheader("💬 Chat")

    if PERSONA:
        st.info(f"🎭 Speaking as **{PERSONA['name']}** — ask anything a recruiter would.")
    else:
        st.warning("Persona not loaded. Answers will be generic.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg and msg["debug"]:
                with st.expander("🔍 Debug"):
                    st.json(msg["debug"])

    if st.session_state.suggested_questions:
        st.write("**💡 Suggestions:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, sq in enumerate(st.session_state.suggested_questions):
            if cols[i].button(sq, key=f"s{i}", use_container_width=True):
                st.session_state.pending_question = sq
                st.rerun()

    if not client:
        st.warning("Set **OPENAI_API_KEY** in your `.env` file to start chatting.")
    else:
        pending = st.session_state.pending_question
        if pending:
            question = pending
            st.session_state.pending_question = None
        else:
            question = st.chat_input("Ask me anything...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    system_prompt = _make_system_prompt()
                    answer, debug_info = handle_question(question, client, system_prompt)
                    st.markdown(answer)
                    with st.expander("🔍 Debug"):
                        st.json(debug_info)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "debug": debug_info,
            })
            st.rerun()
