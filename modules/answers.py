import re
import json
from typing import List, Dict, Optional

from openai import OpenAI

from modules.config import CHAT_MODEL, ANSWER_MODEL
from modules.prompts import GuardedPrompts
from modules.utils import get_document_display_name
from modules.date_utils import format_date_tuple, filter_entries_by_date
from modules.memory import format_short_term_memory_for_prompt, format_user_facts_for_prompt

# =============================================================================
# SINGLE-TENANT ANSWER GENERATION
# =============================================================================


def _build_document_block(documents: List[Dict], time_period=None) -> str:
    """Aggregate all documents owned by the single user into one context block."""
    parts: List[str] = []
    for doc in documents:
        label = get_document_display_name(doc)
        doc_type = doc.get("detected_type", doc.get("file_type", "document"))
        header = f"=== {label} ({doc_type}) ==="

        text_excerpt = doc["text"][:3000]

        wh_block = ""
        work_history = doc.get("work_history", [])
        if work_history:
            entries = work_history
            if time_period:
                filtered = filter_entries_by_date(entries, time_period)
                if filtered:
                    entries = filtered
            wh_lines = []
            for e in entries[:10]:
                wh_lines.append(
                    f"  - {e.get('title', 'Role')} at {e.get('organization', '?')} "
                    f"({e.get('start_display', '?')} – {e.get('end_display', '?')})"
                )
            wh_block = "\nWork/Education history:\n" + "\n".join(wh_lines)

        parts.append(f"{header}\n{text_excerpt}{wh_block}")

    return "\n\n".join(parts)


def _format_conversation_context(conversation_context: list) -> str:
    """Turn the rolling context window into a readable string for the prompt."""
    if not conversation_context:
        return "(start of conversation)"
    lines = []
    for turn in conversation_context[-5:]:
        lines.append(f"Recruiter: {turn.get('question', '')}")
        if turn.get("answer_snippet"):
            lines.append(f"You: {turn['answer_snippet']}")
    return "\n".join(lines)


def answer_as_persona(
    question: str,
    documents: List[Dict],
    system_prompt: str,
    conversation_context: list,
    client: OpenAI,
    time_period=None,
    short_term_memory: Optional[Dict] = None,
    user_facts_memory: Optional[Dict] = None,
) -> str:
    """Generate an answer speaking as the persona (single-tenant)."""
    if not client:
        return "API key required."

    doc_block = _build_document_block(documents, time_period)
    ctx_block = (
        format_short_term_memory_for_prompt(short_term_memory)
        if short_term_memory
        else _format_conversation_context(conversation_context)
    )
    facts_block = format_user_facts_for_prompt(user_facts_memory)

    user_prompt = GuardedPrompts.ANSWER_GENERATION_USER.format(
        question=question,
        conversation_context=ctx_block,
        user_facts_memory=facts_block,
        document_data=doc_block,
    )

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def generate_suggestions(
    question: str,
    answer: str,
    persona_name: str,
    documents: List[Dict],
    client: Optional[OpenAI],
) -> List[str]:
    """Generate follow-up questions a recruiter might ask about the persona."""
    if not client or not documents:
        return []
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are helping a recruiter who is chatting with {persona_name}. "
                        "Generate 3 brief, natural follow-up questions the recruiter "
                        "might ask next. Output only the questions, one per line."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recruiter asked: {question}\n"
                        f"{persona_name} answered: {answer[:300]}\n\n"
                        "Suggest 3 follow-up questions:\n1.\n2.\n3."
                    ),
                },
            ],
            temperature=0.7,
            max_completion_tokens=150,
        )
        suggestions = []
        for line in response.choices[0].message.content.split("\n"):
            cleaned = re.sub(r"^[\d]+[.):]\s*", "", line.strip())
            if cleaned and len(cleaned) > 5:
                suggestions.append(cleaned)
        return suggestions[:3]
    except Exception:
        return []
