"""
Agent tools for the recruiter-facing chatbot.

Five callable tools the agent can invoke during a conversation:
    T1  web_search       – general web lookup (companies, roles, news)
    T2  semantic_search  – meaning-based search over the candidate's documents
    T3  weather_lookup   – current weather for small-talk / logistics
    T4  github_search    – search candidate's GitHub repos & activity
    T5  linkedin_search  – look up professional info via LinkedIn / web

Every tool returns a consistent dict:
    {"results": [...], "source": "<tool_name>", "error": None}      on success
    {"results": [],    "source": "<tool_name>", "error": "<msg>"}   on failure

The TOOL_DEFINITIONS list provides OpenAI function-calling schemas so a
future agent loop can advertise these tools to the model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds for HTTP calls


def _extract_github_username(value: str) -> Optional[str]:
    """Normalize a GitHub handle from URL/username/email-like input."""
    raw = (value or "").strip()
    if not raw:
        return None

    # Handle mistaken forms like "username@github.com".
    if raw.lower().endswith("@github.com"):
        raw = raw.split("@", 1)[0].strip()

    # Handle raw username (with/without leading @).
    if "github.com" not in raw.lower():
        candidate = raw.lstrip("@").strip("/")
        return candidate or None

    # Handle full URL values.
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
        candidate = (parsed.path or "").strip("/").split("/")[0].lstrip("@")
        return candidate or None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — Web Search  (DuckDuckGo, no API key required)
# ═══════════════════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web for companies, roles, news, or any external info.

    Uses the DuckDuckGo search library (no API key needed).
    Falls back to DuckDuckGo Instant-Answer API if the library is unavailable.
    """
    if not query or not query.strip():
        return {"results": [], "source": "web_search", "error": "Empty query"}

    # Primary: ddgs library (successor to duckduckgo-search)
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in raw
        ]
        return {"results": results, "source": "web_search", "error": None}

    except ImportError:
        pass

    # Legacy fallback: duckduckgo-search (older package name)
    try:
        from duckduckgo_search import DDGS as DDGS_Legacy

        with DDGS_Legacy() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in raw
        ]
        return {"results": results, "source": "web_search", "error": None}

    except ImportError:
        logger.warning("Neither ddgs nor duckduckgo-search installed; using instant-answer API")
    except Exception as exc:
        logger.warning("duckduckgo-search failed (%s); trying fallback", exc)

    # Fallback: DuckDuckGo Instant Answer JSON API (limited but zero-dep)
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            })
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:120],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                })

        if not results:
            return {
                "results": [],
                "source": "web_search",
                "error": "No results found",
            }
        return {"results": results[:max_results], "source": "web_search", "error": None}

    except Exception as exc:
        return {"results": [], "source": "web_search", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — Semantic Search  (FAISS vector DB over candidate's documents)
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_search(
    query: str,
    vector_db,
    chunks_metadata: List[Dict],
    embedding_model,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search the candidate's own documents by meaning (not just keywords).

    This is the primary source for factual claims about the candidate — their
    CV, cover letters, transcripts, and any other uploaded documents.

    Parameters
    ----------
    vector_db : faiss.Index
        The FAISS index built from document chunks.
    chunks_metadata : list[dict]
        Parallel list of metadata dicts (text, owner, doc_id, file_type).
    embedding_model : SentenceTransformer
        The same model used to build the index.
    """
    if not query or not query.strip():
        return {"results": [], "source": "semantic_search", "error": "Empty query"}
    if vector_db is None or not chunks_metadata or embedding_model is None:
        return {
            "results": [],
            "source": "semantic_search",
            "error": "Vector DB not initialised (no documents loaded yet)",
        }

    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = min(top_k, len(chunks_metadata))
        distances, indices = vector_db.search(query_embedding, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(chunks_metadata):
                continue
            meta = chunks_metadata[idx]
            results.append({
                "rank": rank + 1,
                "text": meta.get("text", ""),
                "owner": meta.get("owner", ""),
                "doc_id": meta.get("doc_id"),
                "file_type": meta.get("file_type", "unknown"),
                "distance": float(dist),
            })

        return {"results": results, "source": "semantic_search", "error": None}

    except Exception as exc:
        return {"results": [], "source": "semantic_search", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — Weather Lookup  (wttr.in — free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

def weather_lookup(location: str) -> Dict[str, Any]:
    """Get current weather for a location (small-talk / logistics).

    Uses wttr.in which requires no API key and returns structured JSON.
    """
    if not location or not location.strip():
        return {"results": [], "source": "weather_lookup", "error": "No location provided"}

    try:
        url = f"https://wttr.in/{requests.utils.quote(location.strip())}"
        resp = requests.get(
            url,
            params={"format": "j1"},
            headers={"User-Agent": "curl/7.68.0"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]

        area_name = (area.get("areaName", [{}])[0].get("value", location))
        country = (area.get("country", [{}])[0].get("value", ""))
        region = (area.get("region", [{}])[0].get("value", ""))

        result = {
            "location": f"{area_name}, {region}, {country}".strip(", "),
            "temperature_c": current.get("temp_C", ""),
            "temperature_f": current.get("temp_F", ""),
            "feels_like_c": current.get("FeelsLikeC", ""),
            "description": (
                current.get("weatherDesc", [{}])[0].get("value", "")
            ),
            "humidity": current.get("humidity", ""),
            "wind_kmph": current.get("windspeedKmph", ""),
            "wind_dir": current.get("winddir16Point", ""),
            "uv_index": current.get("uvIndex", ""),
            "visibility_km": current.get("visibility", ""),
        }

        return {"results": [result], "source": "weather_lookup", "error": None}

    except Exception as exc:
        return {"results": [], "source": "weather_lookup", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — GitHub Search  (public REST API — 60 req/h unauthenticated)
# ═══════════════════════════════════════════════════════════════════════════════

_GH_API = "https://api.github.com"


def _gh_headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_search(
    query: str,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """Search GitHub for repos, contributions, or tech context.

    If *username* is provided the search is scoped to that user's public repos
    and profile. Otherwise does a general repository search.

    Parameters
    ----------
    query : str
        Free-text query (e.g. "machine learning", "Python projects").
    username : str, optional
        GitHub username to scope the search to.  Falls back to the
        ``github`` field in persona.yaml contact info.
    """
    if not query or not query.strip():
        return {"results": [], "source": "github_search", "error": "Empty query"}

    results: List[Dict[str, Any]] = []

    try:
        # If a username is provided, fetch their profile + repos
        if username:
            # User profile
            profile_resp = requests.get(
                f"{_GH_API}/users/{username}",
                headers=_gh_headers(),
                timeout=_TIMEOUT,
            )
            if profile_resp.ok:
                p = profile_resp.json()
                results.append({
                    "type": "profile",
                    "login": p.get("login", ""),
                    "name": p.get("name", ""),
                    "bio": p.get("bio", ""),
                    "public_repos": p.get("public_repos", 0),
                    "followers": p.get("followers", 0),
                    "html_url": p.get("html_url", ""),
                    "location": p.get("location", ""),
                    "company": p.get("company", ""),
                })

            # Search repos scoped to user
            search_q = f"{query} user:{username}"
            repo_resp = requests.get(
                f"{_GH_API}/search/repositories",
                params={"q": search_q, "sort": "updated", "per_page": 5},
                headers=_gh_headers(),
                timeout=_TIMEOUT,
            )
            if repo_resp.ok:
                for repo in repo_resp.json().get("items", [])[:5]:
                    results.append(_format_repo(repo))
            elif repo_resp.status_code == 422:
                # Fallback: list user repos and filter client-side
                list_resp = requests.get(
                    f"{_GH_API}/users/{username}/repos",
                    params={"sort": "updated", "per_page": 30},
                    headers=_gh_headers(),
                    timeout=_TIMEOUT,
                )
                if list_resp.ok:
                    q_lower = query.lower()
                    for repo in list_resp.json():
                        haystack = (
                            f"{repo.get('name', '')} {repo.get('description', '')} "
                            f"{repo.get('language', '')}".lower()
                        )
                        if q_lower in haystack or any(w in haystack for w in q_lower.split()):
                            results.append(_format_repo(repo))
                    results = results[:5]
        else:
            # General repo search
            repo_resp = requests.get(
                f"{_GH_API}/search/repositories",
                params={"q": query, "sort": "stars", "per_page": 5},
                headers=_gh_headers(),
                timeout=_TIMEOUT,
            )
            if repo_resp.ok:
                for repo in repo_resp.json().get("items", [])[:5]:
                    results.append(_format_repo(repo))

        if not results:
            return {
                "results": [],
                "source": "github_search",
                "error": "No matching repositories or profile found",
            }
        return {"results": results, "source": "github_search", "error": None}

    except Exception as exc:
        return {"results": [], "source": "github_search", "error": str(exc)}


def _format_repo(repo: dict) -> Dict[str, Any]:
    return {
        "type": "repository",
        "name": repo.get("full_name") or repo.get("name", ""),
        "description": repo.get("description", ""),
        "language": repo.get("language", ""),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "url": repo.get("html_url", ""),
        "updated_at": repo.get("updated_at", ""),
        "topics": repo.get("topics", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — LinkedIn Search  (web search scoped to linkedin.com)
# ═══════════════════════════════════════════════════════════════════════════════

def linkedin_search(
    query: str,
    profile_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Look up LinkedIn profile data or professional info.

    LinkedIn has no free public API, so this tool:
      1. If a *profile_url* is given, scrapes the public profile page for
         structured data (name, headline, summary).
      2. Falls back to a web search scoped to ``site:linkedin.com``.

    Parameters
    ----------
    query : str
        What to look for (e.g. "Batuhan Isik experience", "recommendations").
    profile_url : str, optional
        Full LinkedIn profile URL.  Falls back to the ``linkedin`` field in
        persona.yaml contact info.
    """
    if not query or not query.strip():
        return {"results": [], "source": "linkedin_search", "error": "Empty query"}

    results: List[Dict[str, Any]] = []

    # Strategy 1: scrape the public profile page for ld+json metadata
    if profile_url:
        # Always include the explicit profile URL so the model can reference
        # the canonical profile even if scraping/search are blocked.
        results.append({"type": "profile_url", "url": profile_url})
        try:
            page = requests.get(
                profile_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; PersonaBot/1.0; "
                        "+https://github.com/example)"
                    ),
                },
                timeout=_TIMEOUT,
            )
            if page.ok:
                extracted = _extract_linkedin_metadata(page.text, profile_url)
                if extracted:
                    results.append(extracted)
        except Exception as exc:
            logger.debug("LinkedIn profile fetch failed: %s", exc)

    # Strategy 2: web search scoped to linkedin.com, anchored to the provided
    # profile URL when available.
    site_query = f"site:linkedin.com {query}"
    if profile_url:
        site_query = f"{site_query} {profile_url}"
    web_results = web_search(site_query, max_results=5)
    if web_results.get("results"):
        for wr in web_results["results"]:
            results.append({
                "type": "web_result",
                "title": wr.get("title", ""),
                "snippet": wr.get("snippet", ""),
                "url": wr.get("url", ""),
            })

    if not results:
        return {
            "results": [],
            "source": "linkedin_search",
            "error": "No LinkedIn results found",
        }
    return {"results": results, "source": "linkedin_search", "error": None}


def _extract_linkedin_metadata(html: str, url: str) -> Optional[Dict[str, Any]]:
    """Pull structured data from a public LinkedIn profile page.

    LinkedIn embeds JSON-LD (``<script type="application/ld+json">``) in
    public profile pages; we try to parse that first, then fall back to
    <meta> tags.
    """
    record: Dict[str, Any] = {"type": "profile", "url": url}

    # Try JSON-LD
    ld_pattern = re.compile(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        re.DOTALL,
    )
    for match in ld_pattern.finditer(html):
        try:
            blob = json.loads(match.group(1))
            if isinstance(blob, dict) and blob.get("@type") == "Person":
                record["name"] = blob.get("name", "")
                record["headline"] = blob.get("jobTitle", "")
                record["description"] = blob.get("description", "")[:500]
                record["location"] = (
                    blob.get("address", {}).get("addressLocality", "")
                )
                return record
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fallback: <meta> tags
    og_title = _meta_content(html, "og:title")
    og_desc = _meta_content(html, "og:description")
    if og_title:
        record["name"] = og_title
        record["headline"] = og_desc or ""
        return record

    return None


def _meta_content(html: str, prop: str) -> Optional[str]:
    pattern = re.compile(
        rf'<meta\s+[^>]*property="{re.escape(prop)}"[^>]*content="([^"]*)"',
        re.IGNORECASE,
    )
    m = pattern.search(html)
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY  (OpenAI function-calling schema for agent loop integration)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for external information — company backgrounds, "
                "job roles, industry news, or any general fact the candidate's own "
                "documents don't cover. Use this when the recruiter asks about a "
                "company, a competitor, market trends, or anything outside the "
                "candidate's uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Search the candidate's own documents (CV, cover letters, "
                "transcripts, notes) by meaning. This is the PRIMARY source for "
                "any factual claim about the candidate — work history, education, "
                "skills, projects. Always try this tool first when answering "
                "questions about the candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language query describing what to look for "
                            "in the candidate's documents."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "description": (
                "Get current weather for a city or location. Useful for small talk "
                "(\"How's the weather in Boston?\") or logistics (\"Will it rain "
                "for tomorrow's on-site?\")."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Boston', 'London, UK'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": (
                "Search the candidate's GitHub profile and repositories. Use this "
                "when the recruiter asks about the candidate's open-source work, "
                "code contributions, tech stack evidence, or specific projects on "
                "GitHub."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to search for, e.g. 'machine learning', "
                            "'Python projects', 'contributions'."
                        ),
                    },
                    "username": {
                        "type": "string",
                        "description": (
                            "GitHub username to scope the search to. "
                            "Omit for a general search."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "linkedin_search",
            "description": (
                "Look up professional information on LinkedIn — the candidate's "
                "profile, headline, recommendations, or activity. Use this when "
                "the recruiter asks about the candidate's LinkedIn presence, "
                "endorsements, or up-to-date professional headline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for, e.g. 'experience', 'skills', "
                            "'recommendations'."
                        ),
                    },
                    "profile_url": {
                        "type": "string",
                        "description": (
                            "Full LinkedIn profile URL. Omit to fall back to "
                            "the URL in persona.yaml."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def get_tool_map() -> Dict[str, Any]:
    """Return a name → callable mapping for all tools."""
    return {
        "web_search": web_search,
        "semantic_search": semantic_search,
        "weather_lookup": weather_lookup,
        "github_search": github_search,
        "linkedin_search": linkedin_search,
    }


def dispatch_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    vector_db=None,
    chunks_metadata: Optional[List[Dict]] = None,
    embedding_model=None,
    persona: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Automatically injects context-dependent parameters (vector DB for
    semantic_search, GitHub username / LinkedIn URL from persona.yaml).
    This is the single entry-point the agent loop should call.
    """
    tool_map = get_tool_map()
    func = tool_map.get(tool_name)
    if func is None:
        return {
            "results": [],
            "source": tool_name,
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        if tool_name == "semantic_search":
            return func(
                query=arguments.get("query", ""),
                vector_db=vector_db,
                chunks_metadata=chunks_metadata or [],
                embedding_model=embedding_model,
                top_k=int(arguments.get("top_k", 5)),
            )

        if tool_name == "github_search":
            username = arguments.get("username")
            if not username and persona:
                gh_url = persona.get("contact", {}).get("github", "")
                if gh_url:
                    username = _extract_github_username(gh_url)
            return func(query=arguments["query"], username=username)

        if tool_name == "linkedin_search":
            profile_url = arguments.get("profile_url")
            if not profile_url and persona:
                profile_url = persona.get("contact", {}).get("linkedin", "")
            return func(query=arguments["query"], profile_url=profile_url or None)

        if tool_name == "web_search":
            return func(
                query=arguments["query"],
                max_results=int(arguments.get("max_results", 5)),
            )

        if tool_name == "weather_lookup":
            return func(location=arguments["location"])

        return func(**arguments)

    except Exception as exc:
        return {"results": [], "source": tool_name, "error": str(exc)}
