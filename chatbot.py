import streamlit as st
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
import json
from datetime import datetime
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import sys
import os
import signal

# =============================================================================
# APP SHUTDOWN FUNCTIONALITY
# =============================================================================

def shutdown_app():
    """
    Cleanly shut down the Streamlit application.
    This will:
    1. Clear all session state
    2. Free resources (vector DB, models)
    3. Terminate the Python process
    """
    # Clear session state to free memory
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Give user feedback
    st.success("👋 Shutting down application...")
    st.info("You can close this browser tab now.")
    
    # Method 1: Use os._exit() for immediate termination
    # This is the most reliable way to stop Streamlit
    os._exit(0)

def request_shutdown():
    """Set a flag to request shutdown on next rerun"""
    st.session_state.shutdown_requested = True

# Check if shutdown was requested
if st.session_state.get('shutdown_requested', False):
    shutdown_app()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Resume RAG System", page_icon="📄", layout="wide")
st.title("📄 Multi-Resume RAG System")

# =============================================================================
# SIDEBAR WITH EXIT BUTTON
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    
    if OPENAI_API_KEY:
        st.success("✅ API Key Set")
    else:
        st.warning("⚠️ Enter API key")
    
    if 'resumes' in st.session_state and st.session_state.resumes:
        st.divider()
        st.subheader(f"📚 {len(st.session_state.resumes)} Resume(s)")
        for r in st.session_state.resumes:
            with st.expander(r['metadata']['owner']):
                st.write(f"**Entries:** {len(r.get('work_history', []))}")
                for item in r.get('work_history', [])[:3]:
                    st.caption(f"• {item.get('title', 'Role')} ({item.get('start_display', '?')} - {item.get('end_display', '?')})")
    
    # EXIT BUTTON - At the bottom of sidebar
    st.divider()
    st.subheader("🚪 Application Control")
    
    # Confirmation checkbox to prevent accidental clicks
    confirm_exit = st.checkbox("I want to quit the application", key="confirm_exit")
    
    if confirm_exit:
        st.warning("⚠️ This will shut down the app completely!")
        if st.button("🛑 Quit Application", type="primary", use_container_width=True):
            request_shutdown()
            st.rerun()
    else:
        st.button("🛑 Quit Application", disabled=True, use_container_width=True)
        st.caption("Check the box above to enable")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# =============================================================================
# SESSION STATE
# =============================================================================
defaults = {
    'resumes': [], 'processed': False, 'chat_history': [],
    'current_focus_person': None, 'conversation_context': [],
    'suggested_questions': [], 'vector_db': None,
    'chunks_metadata': [], 'embedding_model': None, 'pending_question': None,
    'shutdown_requested': False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# DATE UTILITIES
# =============================================================================

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
    'spring': 3, 'summer': 6, 'fall': 9, 'autumn': 9, 'winter': 12
}

def parse_date_to_yyyymm(date_str):
    if not date_str:
        return None
    date_str = date_str.strip().lower()
    if date_str in ['present', 'current', 'now', 'ongoing', 'today']:
        now = datetime.now()
        return (now.year, now.month)
    for month_name, month_num in MONTH_MAP.items():
        if month_name in date_str:
            year_match = re.search(r'(20\d{2}|19\d{2})', date_str)
            if year_match:
                return (int(year_match.group(1)), month_num)
    match = re.match(r'(\d{1,2})[/-](\d{4})', date_str)
    if match:
        return (int(match.group(2)), int(match.group(1)))
    match = re.match(r'(\d{4})[/-](\d{1,2})', date_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    match = re.match(r'^(20\d{2}|19\d{2})$', date_str)
    if match:
        return (int(match.group(1)), 1)
    try:
        parsed = date_parser.parse(date_str, fuzzy=True)
        return (parsed.year, parsed.month)
    except:
        pass
    return None

def date_tuple_to_sortable(date_tuple):
    if not date_tuple:
        return None
    return date_tuple[0] * 100 + date_tuple[1]

def is_date_in_range(query_date, start_date, end_date):
    if not query_date:
        return False
    query_val = date_tuple_to_sortable(query_date)
    start_val = date_tuple_to_sortable(start_date) if start_date else 0
    end_val = date_tuple_to_sortable(end_date) if end_date else 999912
    return start_val <= query_val <= end_val

def format_date_tuple(date_tuple):
    if not date_tuple:
        return "Unknown"
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return f"{months[date_tuple[1]]} {date_tuple[0]}"

def extract_query_date(question):
    q = question.lower()
    for month_name, month_num in MONTH_MAP.items():
        if month_name in q:
            year_match = re.search(r'(20\d{2}|19\d{2})', q)
            if year_match:
                return (int(year_match.group(1)), month_num)
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', q)
    if year_match:
        return (int(year_match.group(1)), None)
    return None

def filter_entries_by_date(work_history, query_date):
    if not work_history or not query_date:
        return work_history or []
    query_year, query_month = query_date
    matching = []
    for entry in work_history:
        start = entry.get('start_parsed')
        end = entry.get('end_parsed')
        if query_month:
            if is_date_in_range((query_year, query_month), start, end):
                matching.append(entry)
        else:
            year_start = (query_year, 1)
            year_end = (query_year, 12)
            start_val = date_tuple_to_sortable(start) if start else 0
            end_val = date_tuple_to_sortable(end) if end else 999912
            year_start_val = date_tuple_to_sortable(year_start)
            year_end_val = date_tuple_to_sortable(year_end)
            if not (end_val < year_start_val or start_val > year_end_val):
                matching.append(entry)
    return matching

# =============================================================================
# TEXT PROCESSING
# =============================================================================

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, num_chunks=5):
    sentences = sent_tokenize(text)
    if len(sentences) < num_chunks:
        return sentences if sentences else [text]
    per_chunk = len(sentences) // num_chunks
    remainder = len(sentences) % num_chunks
    chunks, start = [], 0
    for i in range(num_chunks):
        size = per_chunk + (1 if i < remainder else 0)
        chunks.append(' '.join(sentences[start:start+size]))
        start += size
    return chunks

def extract_keywords(question):
    remove = [r'\b(who|what|does|do|has|have|had|is|are|was|were|can|could|any|anyone)\b',
              r'\b(experience|skill|knowledge|work|worked|doing|did)\b',
              r'\b(with|in|at|on|for|the|a|an|of|to|and|or|about|tell|me|during)\b', r'[?.,!]']
    cleaned = question.lower()
    for p in remove:
        cleaned = re.sub(p, ' ', cleaned)
    return [w.strip() for w in cleaned.split() if len(w.strip()) > 2]

def keyword_search(keywords, resumes):
    results = {}
    for resume in resumes:
        owner = resume['metadata']['owner']
        text_lower = resume['text'].lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                if owner not in results:
                    results[owner] = {'keywords': [], 'contexts': []}
                if kw not in results[owner]['keywords']:
                    results[owner]['keywords'].append(kw)
                    idx = text_lower.find(kw.lower())
                    start, end = max(0, idx-80), min(len(resume['text']), idx+80)
                    results[owner]['contexts'].append(f"...{resume['text'][start:end]}...")
    return results

# =============================================================================
# VECTOR DB
# =============================================================================

def create_embeddings(texts, model=None):
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=False), model

def build_vector_db(resumes):
    all_chunks, chunks_meta = [], []
    for resume in resumes:
        owner = resume['metadata']['owner']
        for idx, chunk in enumerate(resume['chunks']):
            all_chunks.append(chunk)
            chunks_meta.append({'text': chunk, 'owner': owner, 'chunk_idx': idx})
    embeddings, model = create_embeddings(all_chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks_meta, model

def semantic_search(query, vector_db, chunks_meta, embed_model, target_person=None, k=5):
    query_emb = embed_model.encode([query]).astype('float32')
    search_k = min(k * 4 if target_person else k * 2, len(chunks_meta))
    distances, indices = vector_db.search(query_emb, search_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        chunk = chunks_meta[idx]
        if target_person and target_person.lower() not in chunk['owner'].lower():
            continue
        results.append({'text': chunk['text'], 'owner': chunk['owner'], 'distance': float(dist)})
        if len(results) >= k:
            break
    return results

# =============================================================================
# RESUME PROCESSING
# =============================================================================

def extract_work_history(text, owner_name):
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Extract ALL work and education entries as JSON. 
Be thorough - include internships, part-time jobs, projects, education.
For dates, use format "Month Year" (e.g., "September 2022", "March 2024").
For ongoing roles, use "Present" as end_date."""},
                {"role": "user", "content": f"""Extract work/education history from this resume as JSON array:
[{{"type": "work|education|project", "organization": "...", "title": "...", "start_date": "Month Year", "end_date": "Month Year or Present", "description": "brief description"}}]

Resume:
{text[:5000]}

Return ONLY valid JSON array. Include ALL entries you can find."""}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
        entries = json.loads(result)
        for entry in entries:
            start_str = entry.get('start_date', '')
            end_str = entry.get('end_date', '')
            entry['start_parsed'] = parse_date_to_yyyymm(start_str)
            entry['end_parsed'] = parse_date_to_yyyymm(end_str)
            entry['start_display'] = format_date_tuple(entry['start_parsed']) if entry['start_parsed'] else start_str
            entry['end_display'] = format_date_tuple(entry['end_parsed']) if entry['end_parsed'] else end_str
        return entries
    except:
        return []

def generate_metadata(text, filename):
    if not client:
        return {"owner": filename, "summary": "No API key"}
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract name and summary from resume."},
                {"role": "user", "content": f"Resume:\n{text[:3000]}\n\nReturn:\nNAME: [full name]\nSUMMARY: [2-3 sentence summary]"}
            ],
            temperature=0.2,
            max_tokens=200
        )
        name, summary = filename, "No summary"
        for line in response.choices[0].message.content.split('\n'):
            if line.startswith('NAME:'):
                name = line.replace('NAME:', '').strip()
            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
        return {"owner": name, "summary": summary}
    except:
        return {"owner": filename, "summary": "Error"}

def process_resume(uploaded_file, index):
    reader = PdfReader(uploaded_file)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    cleaned = clean_text(text)
    metadata = generate_metadata(cleaned, uploaded_file.name)
    metadata['num_pages'] = len(reader.pages)
    chunks = chunk_text(cleaned, 5)
    work_history = extract_work_history(cleaned, metadata['owner'])
    return {
        'name': uploaded_file.name,
        'text': cleaned,
        'chunks': chunks,
        'metadata': metadata,
        'work_history': work_history,
        'index': index
    }

# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

def classify_question(question, available_people, current_focus):
    if not client:
        return fallback_classify(question, available_people, current_focus)
    people_str = ", ".join(available_people)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Classify resume questions. Return JSON.
Intents: list_all, specific_person, general_search, timeline, comparison"""},
                {"role": "user", "content": f"""Question: "{question}"
Available: {people_str}
Currently discussing: {current_focus or "no one"}

Rules:
1. Pronouns (he/she/his/her/they) + current focus → target that person
2. Name mentioned → target that person  
3. "who has X", "anyone with X" → general_search
4. Year/month/date mentioned → timeline
5. "all candidates", "whose resumes" → list_all

Return: {{"intent": "...", "target_person": "name or null", "time_period": "raw time string or null"}}"""}
            ],
            temperature=0.1,
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
        return json.loads(result)
    except:
        return fallback_classify(question, available_people, current_focus)

def fallback_classify(question, available_people, current_focus):
    q = question.lower()
    list_patterns = ['who do you have', 'whose resume', 'list all', 'all candidates', 'how many', 'show all', 'everyone', 'available']
    if any(p in q for p in list_patterns):
        return {"intent": "list_all", "target_person": None, "time_period": None}
    for person in available_people:
        name_lower = person.lower()
        first = name_lower.split()[0] if name_lower else ""
        if name_lower in q or (first and len(first) > 2 and f" {first}" in f" {q}"):
            query_date = extract_query_date(question)
            return {"intent": "timeline" if query_date else "specific_person", "target_person": person, "time_period": question if query_date else None}
    pronouns = [' he ', ' she ', ' his ', ' her ', ' him ', ' they ', ' their ']
    if any(p in f" {q} " for p in pronouns) and current_focus:
        query_date = extract_query_date(question)
        return {"intent": "timeline" if query_date else "specific_person", "target_person": current_focus, "time_period": question if query_date else None}
    general = ['who has', 'who have', 'who is', 'who are', 'anyone', 'does any']
    if any(p in q for p in general):
        return {"intent": "general_search", "target_person": None, "time_period": None}
    return {"intent": "general_search", "target_person": None, "time_period": None}

# =============================================================================
# ANSWER GENERATION
# =============================================================================

def answer_list_all(resumes):
    if not resumes:
        return "No resumes uploaded."
    lines = [f"I have **{len(resumes)} candidate(s)** loaded:\n"]
    for i, r in enumerate(resumes, 1):
        lines.append(f"**{i}. {r['metadata']['owner']}**")
        lines.append(f"   _{r['metadata']['summary']}_\n")
    return "\n".join(lines)

def answer_timeline_question(question, person_name, resumes, time_period_str):
    if not client:
        return "API key required"
    target = None
    for r in resumes:
        if person_name.lower() in r['metadata']['owner'].lower():
            target = r
            break
    if not target:
        return f"Couldn't find '{person_name}'. Available: {', '.join([r['metadata']['owner'] for r in resumes])}"
    owner = target['metadata']['owner']
    query_date = extract_query_date(time_period_str or question)
    if not query_date:
        return f"I couldn't understand the time period. Please specify like 'June 2024' or '2023'."
    query_year, query_month = query_date
    query_display = f"{format_date_tuple(query_date)}" if query_month else f"year {query_year}"
    all_entries = target.get('work_history', [])
    matching_entries = filter_entries_by_date(all_entries, query_date)
    if matching_entries:
        entries_text = f"\n✅ ENTRIES THAT OVERLAP WITH {query_display.upper()}:\n"
        for e in matching_entries:
            entries_text += f"\n• **{e.get('title', 'Role')}** at {e.get('organization', 'Organization')}\n  Period: {e.get('start_display', '?')} to {e.get('end_display', '?')}\n  Description: {e.get('description', 'N/A')}\n"
    else:
        entries_text = f"\n❌ NO ENTRIES FOUND that overlap with {query_display}.\n\nAll entries in this resume:\n"
        for e in all_entries:
            entries_text += f"• {e.get('title', 'Role')} at {e.get('organization', 'Org')}: {e.get('start_display', '?')} - {e.get('end_display', '?')}\n"
    prompt = f"""Question: "{question}"

QUERY TIME PERIOD: {query_display}
PERSON: {owner}

{entries_text}

FULL RESUME TEXT (for additional context):
{target['text'][:3000]}

===== INSTRUCTIONS =====
1. The user asked about {query_display}
2. I have PRE-FILTERED the work history above to show ONLY entries that overlap with {query_display}
3. If there are matching entries (✅), describe what {owner} was doing during {query_display}
4. If there are NO matching entries (❌), say: "Based on {owner}'s resume, I cannot find any role during {query_display}."
5. ONLY discuss {owner} - never mention other people"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Answer timeline questions about {owner}. Trust the pre-filtered entries."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_specific_person(question, person_name, resumes):
    if not client:
        return "API key required"
    target = None
    for r in resumes:
        if person_name.lower() in r['metadata']['owner'].lower():
            target = r
            break
    if not target:
        return f"Couldn't find '{person_name}'."
    owner = target['metadata']['owner']
    keywords = extract_keywords(question)
    kw_results = keyword_search(keywords, [target])
    kw_ctx = ""
    if owner in kw_results:
        kw_ctx = f"\nKeyword matches for '{', '.join(keywords)}':\n"
        for ctx in kw_results[owner]['contexts'][:3]:
            kw_ctx += f"• {ctx}\n"
    work_ctx = ""
    if target.get('work_history'):
        work_ctx = "\nWork/Education History:\n"
        for e in target['work_history']:
            work_ctx += f"• {e.get('title', 'Role')} at {e.get('organization', 'Org')} ({e.get('start_display', '?')} - {e.get('end_display', '?')})\n"
    prompt = f"""Question: "{question}"

RESUME OF {owner.upper()}:
{target['text']}
{work_ctx}
{kw_ctx}

Answer using ONLY this resume. Do not mention other people."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Answer questions about {owner} only."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_general_search(question, resumes):
    if not client:
        return "API key required"
    keywords = extract_keywords(question)
    kw_results = keyword_search(keywords, resumes)
    kw_summary = ""
    if kw_results:
        kw_summary = "\n📍 KEYWORD MATCHES:\n"
        for owner, data in kw_results.items():
            kw_summary += f"• **{owner}**: Found [{', '.join(data['keywords'])}]\n"
    all_text = []
    for r in resumes:
        owner = r['metadata']['owner']
        work = ""
        if r.get('work_history'):
            work = "\nHistory:\n" + "\n".join([f"  • {e.get('title')} @ {e.get('organization')} ({e.get('start_display', '?')} - {e.get('end_display', '?')})" for e in r['work_history'][:5]])
        all_text.append(f"════ {owner.upper()} ════\n{r['text'][:3000]}{work}\n════════════════════")
    names = [r['metadata']['owner'] for r in resumes]
    prompt = f"""Question: "{question}"
{kw_summary}

ALL {len(resumes)} RESUMES:
{"".join(all_text)}

Check ALL {len(resumes)} candidates: {', '.join(names)}
List ALL matches with evidence."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Search all {len(resumes)} resumes. Check everyone."}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_suggestions(question, answer, target, resumes):
    if not client or not resumes:
        return []
    names = [r['metadata']['owner'] for r in resumes]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Generate 3 follow-up questions."}, {"role": "user", "content": f"Q: {question}\nA: {answer[:200]}\nFocus: {target}\nPeople: {names}\n\n1.\n2.\n3."}],
            temperature=0.7,
            max_tokens=150
        )
        suggestions = []
        for line in response.choices[0].message.content.split('\n'):
            cleaned = re.sub(r'^[\d]+[.):]\s*', '', line.strip())
            if cleaned and len(cleaned) > 5:
                suggestions.append(cleaned)
        return suggestions[:3]
    except:
        return []

# =============================================================================
# MAIN HANDLER
# =============================================================================

def handle_question(question):
    resumes = st.session_state.resumes
    current_focus = st.session_state.current_focus_person
    available = [r['metadata']['owner'] for r in resumes]
    classification = classify_question(question, available, current_focus)
    intent = classification.get('intent', 'general_search')
    target = classification.get('target_person')
    time_period = classification.get('time_period')
    
    if intent == 'list_all':
        answer = answer_list_all(resumes)
        new_focus = None
    elif intent == 'timeline' and target:
        answer = answer_timeline_question(question, target, resumes, time_period)
        new_focus = target
    elif intent == 'specific_person' and target:
        answer = answer_specific_person(question, target, resumes)
        new_focus = target
    else:
        answer = answer_general_search(question, resumes)
        new_focus = None
    
    st.session_state.current_focus_person = new_focus
    st.session_state.conversation_context.append({'question': question, 'intent': intent, 'target': target})
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context.pop(0)
    st.session_state.suggested_questions = generate_suggestions(question, answer, target, resumes)
    return answer, classification

# =============================================================================
# UI - MAIN AREA
# =============================================================================

if not st.session_state.processed:
    st.subheader("📤 Upload Resumes")
    files = st.file_uploader("Choose PDFs", type=['pdf'], accept_multiple_files=True)
    
    if files and st.button("🚀 Process All", type="primary"):
        if not OPENAI_API_KEY:
            st.error("Enter API key first")
        else:
            progress = st.progress(0)
            processed = []
            for i, f in enumerate(files):
                st.text(f"Processing: {f.name}")
                processed.append(process_resume(f, i))
                progress.progress((i+1) / len(files))
            st.text("Building vector database...")
            vector_db, chunks_meta, embed_model = build_vector_db(processed)
            st.session_state.resumes = processed
            st.session_state.vector_db = vector_db
            st.session_state.chunks_metadata = chunks_meta
            st.session_state.embedding_model = embed_model
            st.session_state.processed = True
            st.success(f"✅ Processed {len(processed)} resume(s)!")
            st.rerun()

if st.session_state.processed:
    st.subheader("📚 Candidates")
    cols = st.columns(min(len(st.session_state.resumes), 4))
    for i, r in enumerate(st.session_state.resumes):
        with cols[i % 4]:
            with st.container(border=True):
                st.write(f"**{r['metadata']['owner']}**")
                st.caption(f"{len(r.get('work_history', []))} timeline entries")
    
    col1, col2 = st.columns([1, 4])
    if col1.button("🗑️ Reset Data"):
        # Clear data but don't exit
        for k in ['resumes', 'processed', 'chat_history', 'current_focus_person', 
                  'conversation_context', 'suggested_questions', 'vector_db', 
                  'chunks_metadata', 'embedding_model', 'pending_question']:
            if k in st.session_state:
                st.session_state[k] = defaults.get(k, None)
        st.session_state.processed = False
        st.rerun()
    
    with st.expander("📤 Add More Resumes"):
        new_files = st.file_uploader("More PDFs", type=['pdf'], accept_multiple_files=True, key="add")
        if new_files and st.button("➕ Add"):
            start = len(st.session_state.resumes)
            for i, f in enumerate(new_files):
                st.session_state.resumes.append(process_resume(f, start + i))
            vector_db, chunks_meta, embed_model = build_vector_db(st.session_state.resumes)
            st.session_state.vector_db = vector_db
            st.session_state.chunks_metadata = chunks_meta
            st.session_state.embedding_model = embed_model
            st.rerun()
    
    st.divider()
    st.subheader("💬 Chat")
    
    if st.session_state.current_focus_person:
        st.info(f"📍 Discussing: **{st.session_state.current_focus_person}**")
    else:
        st.info("📍 Searching: **All candidates**")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg and msg["debug"]:
                with st.expander("🔍 Analysis"):
                    st.json(msg["debug"])
    
    if st.session_state.suggested_questions:
        st.write("**💡 Suggestions:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, sq in enumerate(st.session_state.suggested_questions):
            if cols[i].button(sq, key=f"s{i}", use_container_width=True):
                st.session_state.pending_question = sq
                st.rerun()
    
    pending = st.session_state.pending_question
    if pending:
        question = pending
        st.session_state.pending_question = None
    else:
        question = st.chat_input("Ask about the resumes...")
    
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, classification = handle_question(question)
                st.markdown(answer)
                with st.expander("🔍 Analysis"):
                    st.json(classification)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "debug": classification})
        st.rerun()