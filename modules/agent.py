"""
Agent loop: reason → tool-call → synthesize.

Each user message is processed by an iterative loop that:
  1. Builds a message list (system prompt + memory + conversation)
  2. Calls the LLM with tool definitions
  3. If the LLM requests tool calls → executes them, appends results, loops
  4. If the LLM returns plain text → returns it as the final persona answer

Capped at MAX_TOOL_ROUNDS to prevent runaway chains.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from modules.config import ANSWER_MODEL
from modules.tools import TOOL_DEFINITIONS
from modules.memory import (
    format_short_term_memory_for_prompt,
    format_user_facts_for_prompt,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6

_AGENT_INSTRUCTIONS = """

AGENT BEHAVIOUR — TOOL USE AND ANSWER SYNTHESIS:

You have access to five tools. Follow these rules for EVERY recruiter message:

DECISION FLOW:
1. Read the question carefully. Decide what information you need.
2. If the recruiter asks about YOUR experience, skills, projects, education, \
or anything personal — call semantic_search FIRST, even if your memory or \
bio seem to cover it. Your documents are the ground truth; memory is a cache.
3. If you need EXTERNAL facts (a company the recruiter mentions, job market \
data, industry news, something not about you) — call web_search.
4. If the recruiter asks about your GitHub work, open-source contributions, \
or repos — call github_search. Your persona config provides the username \
automatically.
5. If they ask about your LinkedIn profile, recommendations, or headline — \
call linkedin_search. Your persona config provides the URL automatically.
6. For weather or logistics small-talk ("What's the weather in Boston?") — \
call weather_lookup.
7. If memory and stored facts alone clearly cover the question (e.g. a simple \
greeting, "what's your name?", or following up on something just discussed), \
reply directly without a tool call.

SYNTHESIS RULES:
- After receiving tool results, compose a natural, first-person answer.
- NEVER dump raw JSON or tool output to the recruiter.
- Blend information from multiple tool results when appropriate.
- If a tool returns no results or an error, do NOT hallucinate — say you don't \
have that information and offer what you do know.
- Stay in character at all times. You are the candidate, not an assistant.
- Be concise but thorough. A recruiter's time is valuable.
- You may call multiple tools in a single round or across rounds.
"""


# ─── message construction ────────────────────────────────────────────────────

def _build_initial_messages(
    system_prompt: str,
    question: str,
    short_term_memory: Optional[Dict],
    user_facts_memory: Optional[Dict],
) -> List[Dict[str, Any]]:
    """Compose the opening message list for the agent loop.

    The *system_prompt* already contains the persona identity, rules, and
    document summaries (built by ``build_system_prompt`` in persona.py).
    We append agent-specific tool-use instructions and pack conversation
    memory + stored facts into the first user message alongside the
    recruiter's question.
    """
    memory_text = format_short_term_memory_for_prompt(short_term_memory)
    facts_text = format_user_facts_for_prompt(user_facts_memory)

    full_system = system_prompt + _AGENT_INSTRUCTIONS

    context_msg = (
        f"[Conversation memory]\n{memory_text}\n\n"
        f"[Stored facts about you]\n{facts_text}\n\n"
        f"---\nRecruiter: {question}"
    )

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": context_msg},
    ]


# ─── core loop ───────────────────────────────────────────────────────────────

def run_agent_loop(
    *,
    question: str,
    system_prompt: str,
    client: OpenAI,
    call_tool: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    short_term_memory: Optional[Dict] = None,
    user_facts_memory: Optional[Dict] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Execute the agent loop for a single recruiter message.

    Returns
    -------
    (answer_text, debug_info)
        *answer_text* is the persona's final response.
        *debug_info* contains tool-call traces and round counts.
    """
    if not client:
        return "API key required.", {"error": "no_client"}

    messages = _build_initial_messages(
        system_prompt, question, short_term_memory, user_facts_memory,
    )

    tool_trace: List[Dict[str, Any]] = []

    for round_idx in range(MAX_TOOL_ROUNDS):
        try:
            response = client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=1024,
            )
        except Exception as exc:
            logger.exception("Agent LLM call failed on round %d", round_idx)
            return f"Error contacting the model: {exc}", {
                "error": str(exc),
                "agent_rounds": round_idx + 1,
                "tool_calls": tool_trace,
            }

        choice = response.choices[0]
        assistant_msg = choice.message

        # ── No tool calls → final answer ─────────────────────────────
        if not assistant_msg.tool_calls:
            answer = assistant_msg.content or ""
            return answer, {
                "agent_rounds": round_idx + 1,
                "tool_calls": tool_trace,
                "finish_reason": choice.finish_reason,
            }

        # ── Tool-call round ──────────────────────────────────────────
        assistant_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ],
        }
        messages.append(assistant_dict)

        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            logger.info("Agent round %d → %s(%s)", round_idx, fn_name, fn_args)
            result = call_tool(fn_name, fn_args)

            tool_trace.append({
                "round": round_idx,
                "tool": fn_name,
                "args": fn_args,
                "result_count": len(result.get("results", [])),
                "error": result.get("error"),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

        # Loop back — the model will see tool results and decide next step.

    # ── Safety net: all rounds exhausted → force a text-only completion ───
    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
    except Exception as exc:
        answer = f"I ran into an issue pulling that together — {exc}"

    return answer, {
        "agent_rounds": MAX_TOOL_ROUNDS + 1,
        "tool_calls": tool_trace,
        "finish_reason": "max_rounds_exceeded",
    }
