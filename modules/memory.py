from __future__ import annotations

import re
from typing import Dict, List, Optional


# =============================================================================
# MEMORY LAYER HELPERS
# =============================================================================


def _clean_text(value: str, limit: int = 240) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned[:limit]


def _append_unique(target: List[str], value: str, max_items: int = 25) -> None:
    cleaned = _clean_text(value)
    if not cleaned:
        return
    if cleaned not in target:
        target.append(cleaned)
    if len(target) > max_items:
        del target[:-max_items]


def init_short_term_memory(max_recent_turns: int = 6) -> Dict:
    return {
        "max_recent_turns": max_recent_turns,
        "recent_turns": [],
        "summary_points": [],
    }


def update_short_term_memory(memory: Dict, user_message: str, assistant_message: str) -> Dict:
    if not memory:
        memory = init_short_term_memory()

    recent_turns = memory.setdefault("recent_turns", [])
    summary_points = memory.setdefault("summary_points", [])
    max_recent_turns = int(memory.get("max_recent_turns", 6))

    recent_turns.append(
        {
            "user": _clean_text(user_message, limit=500),
            "assistant": _clean_text(assistant_message, limit=700),
        }
    )

    while len(recent_turns) > max_recent_turns:
        moved = recent_turns.pop(0)
        summary_line = (
            f"Recruiter asked about: {moved.get('user', '')}. "
            f"You answered: {moved.get('assistant', '')[:160]}"
        )
        _append_unique(summary_points, summary_line, max_items=15)

    return memory


def format_short_term_memory_for_prompt(memory: Optional[Dict]) -> str:
    if not memory:
        return "(start of conversation)"

    lines: List[str] = []
    summary_points = memory.get("summary_points", [])
    recent_turns = memory.get("recent_turns", [])

    if summary_points:
        lines.append("Earlier context summary:")
        for point in summary_points[-6:]:
            lines.append(f"- {point}")
        lines.append("")

    if recent_turns:
        lines.append("Most recent turns:")
        for turn in recent_turns[-6:]:
            lines.append(f"Recruiter: {turn.get('user', '')}")
            lines.append(f"You: {turn.get('assistant', '')}")
    else:
        lines.append("(start of conversation)")

    return "\n".join(lines).strip()


def init_user_facts_memory(persona: Optional[Dict] = None, documents: Optional[List[Dict]] = None) -> Dict:
    facts: Dict = {
        "profile": {},
        "skills": [],
        "roles": [],
        "organizations": [],
        "preferences": [],
        "key_facts": [],
        "source_documents": [],
    }

    if persona:
        for key in ("name", "role", "location"):
            value = persona.get(key)
            if value:
                facts["profile"][key] = _clean_text(str(value), limit=120)

        bio = persona.get("bio", "")
        if bio:
            _append_unique(facts["key_facts"], f"Bio: {_clean_text(bio, limit=260)}")

    for doc in documents or []:
        doc_name = doc.get("name", "document")
        _append_unique(facts["source_documents"], doc_name, max_items=100)

        metadata = doc.get("metadata", {})
        summary = metadata.get("summary")
        if summary:
            _append_unique(facts["key_facts"], summary)

        for entry in doc.get("work_history", [])[:30]:
            role = entry.get("title")
            org = entry.get("organization")
            if role:
                _append_unique(facts["roles"], role)
            if org:
                _append_unique(facts["organizations"], org)

            start = entry.get("start_display", "?")
            end = entry.get("end_display", "?")
            if role or org:
                _append_unique(
                    facts["key_facts"],
                    f"{role or 'Role'} at {org or 'organization'} ({start} - {end})",
                )

            description = entry.get("description")
            if description:
                _append_unique(facts["key_facts"], description)

        extracted = doc.get("extracted_data", {})
        for item in extracted.get("key_entities", [])[:15]:
            _append_unique(facts["organizations"], str(item))
        for item in extracted.get("key_facts", [])[:20]:
            _append_unique(facts["key_facts"], str(item))

        # Lightweight skill mining from text snippets.
        text = (doc.get("text") or "")[:8000]
        for match in re.findall(r"\b([A-Z][A-Za-z0-9+#.\-]{1,24})\b", text):
            if match.lower() in {"i", "we", "the", "and", "for", "with"}:
                continue
            if match.isupper() and len(match) <= 5:
                _append_unique(facts["skills"], match)

    return facts


def update_user_facts_from_explicit_message(user_facts: Dict, message: str) -> Dict:
    if not user_facts:
        user_facts = init_user_facts_memory()

    msg = (message or "").strip()
    if not msg:
        return user_facts

    lowered = msg.lower()
    if lowered.startswith("remember that ") or "for future reference" in lowered:
        _append_unique(user_facts.setdefault("key_facts", []), msg, max_items=40)

    if lowered.startswith("you prefer") or "your preference is" in lowered:
        _append_unique(user_facts.setdefault("preferences", []), msg, max_items=20)

    return user_facts


def format_user_facts_for_prompt(user_facts: Optional[Dict]) -> str:
    if not user_facts:
        return "(no stored user facts yet)"

    profile = user_facts.get("profile", {})
    lines: List[str] = []

    if profile:
        pieces = [f"{k}: {v}" for k, v in profile.items() if v]
        if pieces:
            lines.append("Profile: " + " | ".join(pieces))

    def _emit(title: str, values: List[str], cap: int = 8) -> None:
        if values:
            lines.append(f"{title}:")
            for value in values[-cap:]:
                lines.append(f"- {value}")

    _emit("Skills", user_facts.get("skills", []))
    _emit("Roles", user_facts.get("roles", []))
    _emit("Organizations", user_facts.get("organizations", []))
    _emit("Preferences", user_facts.get("preferences", []), cap=6)
    _emit("Key facts", user_facts.get("key_facts", []), cap=10)

    if not lines:
        return "(no stored user facts yet)"
    return "\n".join(lines)
