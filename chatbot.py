
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
import hashlib
from datetime import datetime
from dateutil import parser as date_parser
import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
CHAT_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"
EXTRACTION_MODEL = "gpt-4o-mini"

# =============================================================================
# FILE SIZE GUARDRAIL
# =============================================================================
MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def check_file_size(uploaded_file):
    if uploaded_file is None:
        return False, 0, "No file"
    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    size_mb = file_size / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, size_mb, f"File exceeds {MAX_FILE_SIZE_MB}MB limit"
    return True, size_mb, None

# =============================================================================
# DUPLICATE DETECTION SYSTEM - FIX FOR ISSUE #2
# =============================================================================

class DuplicateDetector:
    """
    Multi-layer duplicate detection for resume uploads.
    Checks: file hash, content fingerprint, candidate name.
    """
    
    @staticmethod
    def compute_file_hash(uploaded_file):
        """Compute SHA-256 hash of raw file bytes"""
        uploaded_file.seek(0)
        file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
        uploaded_file.seek(0)
        return file_hash
    
    @staticmethod
    def compute_content_fingerprint(text):
        """
        Create normalized fingerprint from resume text.
        Resistant to minor formatting changes.
        """
        # Normalize: lowercase, alphanumeric only
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        # Sort unique words and take sample for fingerprint
        words = sorted(set(normalized.split()))
        # Use first 150 words as fingerprint base (captures key terms)
        fingerprint_base = ' '.join(words[:150])
        return hashlib.sha256(fingerprint_base.encode()).hexdigest()
    
    @staticmethod
    def normalize_name(name):
        """Normalize candidate name for comparison"""
        # Remove titles, punctuation, extra spaces
        name = re.sub(r'\b(mr|mrs|ms|dr|prof|jr|sr|ii|iii|iv)\b\.?', '', name.lower())
        name = re.sub(r'[^\w\s]', '', name)
        return ' '.join(name.split())
    
    @staticmethod
    def name_similarity(name1, name2):
        """Calculate name similarity score (0-1)"""
        n1_parts = set(DuplicateDetector.normalize_name(name1).split())
        n2_parts = set(DuplicateDetector.normalize_name(name2).split())
        
        if not n1_parts or not n2_parts:
            return 0.0
        
        intersection = len(n1_parts & n2_parts)
        union = len(n1_parts | n2_parts)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def check_duplicate(uploaded_file, extracted_text, candidate_name, existing_resumes):
        """
        Comprehensive duplicate check.
        Returns: (is_duplicate, duplicate_type, existing_candidate, message)
        
        Duplicate types:
        - 'exact_file': Identical PDF file
        - 'content_match': Same content, different file
        - 'name_match': Same candidate name
        - 'name_similar': Similar candidate name (potential update)
        """
        if not existing_resumes:
            return False, None, None, None
        
        # Layer 1: Exact file hash
        new_file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
        for resume in existing_resumes:
            if resume.get('file_hash') == new_file_hash:
                return True, 'exact_file', resume['metadata']['owner'], \
                    f"This exact file was already uploaded for {resume['metadata']['owner']}"
        
        # Layer 2: Content fingerprint
        new_fingerprint = DuplicateDetector.compute_content_fingerprint(extracted_text)
        for resume in existing_resumes:
            if resume.get('content_fingerprint') == new_fingerprint:
                return True, 'content_match', resume['metadata']['owner'], \
                    f"This resume has identical content to {resume['metadata']['owner']}'s resume"
        
        # Layer 3: Candidate name (exact)
        new_name_norm = DuplicateDetector.normalize_name(candidate_name)
        for resume in existing_resumes:
            existing_name_norm = DuplicateDetector.normalize_name(resume['metadata']['owner'])
            if new_name_norm == existing_name_norm:
                return True, 'name_match', resume['metadata']['owner'], \
                    f"A resume for {resume['metadata']['owner']} already exists. Upload as update?"
        
        # Layer 4: Candidate name (similar - warn but allow)
        for resume in existing_resumes:
            similarity = DuplicateDetector.name_similarity(candidate_name, resume['metadata']['owner'])
            if similarity >= 0.6:  # 60% name overlap
                return True, 'name_similar', resume['metadata']['owner'], \
                    f"Similar to existing candidate {resume['metadata']['owner']}. Is this the same person?"
        
        return False, None, None, None


# =============================================================================
# COMPLETE SESSION RESET - FIX FOR ISSUE #1
# =============================================================================

def get_default_state():
    """Returns a fresh default state dictionary"""
    return {
        'resumes': [],
        'processed': False,
        'chat_history': [],
        'current_focus_person': None,
        'last_mentioned_person': None,
        'last_response_entity': None,  # NEW: Track entity from last bot response
        'conversation_context': [],
        'suggested_questions': [],
        'vector_db': None,
        'chunks_metadata': [],
        'embedding_model': None,
        'pending_question': None,
        'shutdown_requested': False,
        'file_rejected': False,
        'reset_counter': 0,
        'duplicate_warnings': [],  # NEW: Track duplicate warnings
        'pending_duplicate': None  # NEW: Store pending duplicate for user decision
    }

def perform_complete_reset():
    """Complete reset - clears ALL state"""
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        del st.session_state[key]
    
    defaults = get_default_state()
    for key, value in defaults.items():
        st.session_state[key] = value
    
    st.session_state.reset_counter = st.session_state.get('reset_counter', 0) + 1
    
    import gc
    gc.collect()
    return True

# =============================================================================
# ENHANCED ENTITY TRACKING - FIX FOR ISSUE #1
# =============================================================================

class EntityTracker:
    """
    Enhanced entity tracking with response-aware context.
    Tracks entities in both questions AND answers.
    """
    
    @staticmethod
    def extract_mentioned_person(text, available_people):
        """Extract explicitly mentioned person from text"""
        text_lower = text.lower()
        for person in available_people:
            name_lower = person.lower()
            if name_lower in text_lower:
                return person
            first_name = name_lower.split()[0] if name_lower else ""
            if first_name and len(first_name) > 2:
                pattern = rf'\b{re.escape(first_name)}\b'
                if re.search(pattern, text_lower):
                    return person
        return None
    
    @staticmethod
    def extract_last_mentioned_person(text, available_people):
        """
        Extract the LAST mentioned person in text (for response tracking).
        Handles cases where multiple people are mentioned.
        """
        text_lower = text.lower()
        last_found = None
        last_position = -1
        
        for person in available_people:
            name_lower = person.lower()
            
            # Find last occurrence of full name
            pos = text_lower.rfind(name_lower)
            if pos > last_position:
                last_position = pos
                last_found = person
            
            # Also check first name
            first_name = name_lower.split()[0] if name_lower else ""
            if first_name and len(first_name) > 2:
                pattern = rf'\b{re.escape(first_name)}\b'
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    last_match_pos = matches[-1].start()
                    if last_match_pos > last_position:
                        last_position = last_match_pos
                        last_found = person
        
        return last_found
    
    @staticmethod
    def count_person_mentions(text, person):
        """Count how many times a person is mentioned"""
        text_lower = text.lower()
        name_lower = person.lower()
        count = text_lower.count(name_lower)
        
        first_name = name_lower.split()[0] if name_lower else ""
        if first_name and len(first_name) > 2:
            pattern = rf'\b{re.escape(first_name)}\b'
            count += len(re.findall(pattern, text_lower))
        
        return count
    
    @staticmethod
    def has_pronoun(text):
        """Check if text contains pronouns that need resolution"""
        text_lower = text.lower()
        pronouns = [
            r'\bhe\b', r'\bshe\b', r'\bhis\b', r'\bher\b', r'\bhim\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\bthis person\b',
            r'\bthat person\b', r'\bthe candidate\b', r'\bthis candidate\b',
            r'\bthat candidate\b'
        ]
        for p in pronouns:
            if re.search(p, text_lower):
                return True
        return False
    
    @staticmethod
    def resolve_target(question, available_people, current_focus, last_mentioned, last_response_entity):
        """
        Resolve who the question is about.
        
        Priority for pronoun resolution:
        1. Last entity mentioned in bot's response (most recent context)
        2. Last person mentioned in user's question
        3. Current focus person
        """
        # Check for explicit mention first
        mentioned = EntityTracker.extract_mentioned_person(question, available_people)
        if mentioned:
            return mentioned, "explicit_mention"
        
        # Check for pronouns
        if EntityTracker.has_pronoun(question):
            # Priority 1: Entity from last bot response
            if last_response_entity and last_response_entity in available_people:
                return last_response_entity, "pronoun_to_response_entity"
            # Priority 2: Last mentioned in conversation
            if last_mentioned and last_mentioned in available_people:
                return last_mentioned, "pronoun_to_last_mentioned"
            # Priority 3: Current focus
            if current_focus and current_focus in available_people:
                return current_focus, "pronoun_to_focus"
        
        return None, "no_target"
    
    @staticmethod
    def update_from_response(response_text, available_people, intent):
        """
        Update tracking state based on bot's response content.
        Called AFTER answer generation.
        """
        # Extract entity from response
        response_entity = EntityTracker.extract_last_mentioned_person(response_text, available_people)
        
        if response_entity:
            st.session_state.last_response_entity = response_entity
            
            # Count mentions to determine if this is a focused response
            mention_count = EntityTracker.count_person_mentions(response_text, response_entity)
            
            # If entity is mentioned multiple times, set as focus
            if mention_count >= 2 or intent in ['specific_person', 'timeline']:
                st.session_state.current_focus_person = response_entity
                st.session_state.last_mentioned_person = response_entity
            else:
                # Single mention in general search - update last_mentioned but not focus
                st.session_state.last_mentioned_person = response_entity
        
        return response_entity
    
    @staticmethod
    def update_from_question(target_person):
        """Update tracking when question explicitly mentions someone"""
        if target_person:
            st.session_state.current_focus_person = target_person
            st.session_state.last_mentioned_person = target_person


# =============================================================================
# QUERY ANALYSIS
# =============================================================================

def requires_all_candidates(question):
    """Detect if query requires iterating over ALL candidates."""
    q_lower = question.lower()
    
    all_patterns = [
        (r'\bwho do you have\b', 'list_all'),
        (r'\bwhose resumes?\b', 'list_all'),
        (r'\blist (all )?(the )?candidates\b', 'list_all'),
        (r'\bshow (all )?(the )?candidates\b', 'list_all'),
        (r'\bhow many (candidates|resumes|people)\b', 'list_all'),
        (r'\ball (the )?(candidates|resumes|people)\b', 'list_all'),
        (r'\beveryone\b', 'list_all'),
        (r'\beverybody\b', 'list_all'),
        (r'\beach (candidate|person|one)\b', 'iterate_all'),
        (r'\bevery (candidate|person|one)\b', 'iterate_all'),
        (r'\bfor all (candidates|people)\b', 'iterate_all'),
        (r'\ball of them\b', 'iterate_all'),
        (r'\bcompare\b', 'iterate_all'),
        (r'\bwhat was .+ doing\b.*\beach\b', 'iterate_all'),
        (r'\beach .+ doing\b', 'iterate_all'),
    ]
    
    for pattern, iteration_type in all_patterns:
        if re.search(pattern, q_lower):
            return True, iteration_type
    
    search_all_patterns = [
        r'\bwho has\b', r'\bwho have\b', r'\bwho is\b', r'\bwho are\b',
        r'\bwho can\b', r'\bwho knows\b', r'\bwho works\b', r'\bwho worked\b',
        r'\banyone with\b', r'\banybody with\b', r'\bany candidate\b'
    ]
    
    for pattern in search_all_patterns:
        if re.search(pattern, q_lower):
            return True, 'search_all'
    
    return False, None

def classify_intent(question, available_people, current_focus, last_mentioned, last_response_entity):
    """Robust intent classification with enhanced entity resolution."""
    
    # Step 1: Check if requires all candidates
    requires_all, all_type = requires_all_candidates(question)
    
    if requires_all:
        if all_type == 'list_all':
            return {
                "intent": "list_all",
                "target_person": None,
                "time_period": None,
                "requires_iteration": False
            }
        elif all_type == 'iterate_all':
            time_period = extract_query_date(question)
            return {
                "intent": "iterate_all",
                "target_person": None,
                "time_period": question if time_period else None,
                "requires_iteration": True
            }
        else:
            return {
                "intent": "general_search",
                "target_person": None,
                "time_period": None,
                "requires_iteration": False
            }
    
    # Step 2: Resolve target person with enhanced tracking
    target, resolution = EntityTracker.resolve_target(
        question, available_people, current_focus, last_mentioned, last_response_entity
    )
    
    # Step 3: Check for timeline query
    time_period = extract_query_date(question)
    
    # Step 4: Determine intent
    if target:
        intent = "timeline" if time_period else "specific_person"
    else:
        intent = "general_search"
    
    return {
        "intent": intent,
        "target_person": target,
        "time_period": question if time_period else None,
        "resolution_method": resolution,
        "requires_iteration": False
    }

# =============================================================================
# APP SHUTDOWN
# =============================================================================

def shutdown_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("👋 Shutting down...")
    os._exit(0)

if st.session_state.get('shutdown_requested', False):
    shutdown_app()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Resume Coach", page_icon="📝", layout="wide")
st.title("📝 Resume Analysis & Coaching Chatbot")

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
defaults = get_default_state()
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    
    if OPENAI_API_KEY:
        st.success("✅ API Key Set")
    else:
        st.warning("⚠️ Enter API key")
    
    st.divider()
    st.subheader("🤖 Models")
    st.caption(f"Chat: {CHAT_MODEL}")
    st.caption(f"Analysis: {ANSWER_MODEL}")
    
    if st.session_state.resumes:
        st.divider()
        st.subheader(f"📚 {len(st.session_state.resumes)} Resume(s)")
        for r in st.session_state.resumes:
            st.write(f"• {r['metadata']['owner']}")
    
    # Debug info - Enhanced
    st.divider()
    st.subheader("🔍 Entity Tracking State")
    st.caption(f"Current focus: {st.session_state.current_focus_person or 'None'}")
    st.caption(f"Last mentioned: {st.session_state.last_mentioned_person or 'None'}")
    st.caption(f"Last response entity: {st.session_state.get('last_response_entity') or 'None'}")
    st.caption(f"Reset count: {st.session_state.reset_counter}")
    
    st.divider()
    confirm_exit = st.checkbox("Quit app", key="confirm_exit")
    if confirm_exit:
        if st.button("🛑 Quit", type="primary"):
            st.session_state.shutdown_requested = True
            st.rerun()

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# =============================================================================
# DATE UTILITIES
# =============================================================================

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}

def parse_date_to_yyyymm(date_str):
    if not date_str:
        return None
    date_str = date_str.strip().lower()
    if date_str in ['present', 'current', 'now', 'ongoing']:
        now = datetime.now()
        return (now.year, now.month)
    for month_name, month_num in MONTH_MAP.items():
        if month_name in date_str:
            year_match = re.search(r'(20\d{2}|19\d{2})', date_str)
            if year_match:
                return (int(year_match.group(1)), month_num)
    year_match = re.search(r'^(20\d{2}|19\d{2})$', date_str)
    if year_match:
        return (int(year_match.group(1)), 1)
    try:
        parsed = date_parser.parse(date_str, fuzzy=True)
        return (parsed.year, parsed.month)
    except:
        return None

def date_tuple_to_sortable(dt):
    return dt[0] * 100 + dt[1] if dt else None

def is_date_in_range(query_date, start_date, end_date):
    if not query_date:
        return False
    qv = date_tuple_to_sortable(query_date)
    sv = date_tuple_to_sortable(start_date) if start_date else 0
    ev = date_tuple_to_sortable(end_date) if end_date else 999912
    return sv <= qv <= ev

def format_date_tuple(dt):
    if not dt:
        return "Unknown"
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return f"{months[dt[1]]} {dt[0]}"

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
            sv = date_tuple_to_sortable(start) if start else 0
            ev = date_tuple_to_sortable(end) if end else 999912
            ysv, yev = query_year * 100 + 1, query_year * 100 + 12
            if not (ev < ysv or sv > yev):
                matching.append(entry)
    return matching

# =============================================================================
# TEXT & VECTOR PROCESSING
# =============================================================================

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', text).strip()

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

def build_vector_db(resumes):
    all_chunks, chunks_meta = [], []
    for resume in resumes:
        owner = resume['metadata']['owner']
        for idx, chunk in enumerate(resume['chunks']):
            all_chunks.append(chunk)
            chunks_meta.append({'text': chunk, 'owner': owner, 'candidate_id': resume.get('candidate_id')})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks_meta, model

# =============================================================================
# RESUME PROCESSING - WITH DUPLICATE DETECTION
# =============================================================================

def extract_work_history(text):
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "Extract work/education as JSON array."},
                {"role": "user", "content": f"""Extract as JSON:
[{{"type":"work|education", "organization":"...", "title":"...", "start_date":"Month Year", "end_date":"Month Year or Present", "description":"brief"}}]

Resume:
{text[:5000]}

Return ONLY JSON."""}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
        entries = json.loads(result)
        for e in entries:
            e['start_parsed'] = parse_date_to_yyyymm(e.get('start_date', ''))
            e['end_parsed'] = parse_date_to_yyyymm(e.get('end_date', ''))
            e['start_display'] = format_date_tuple(e['start_parsed']) or e.get('start_date', '?')
            e['end_display'] = format_date_tuple(e['end_parsed']) or e.get('end_date', '?')
        return entries
    except:
        return []

def generate_metadata(text, filename):
    if not client:
        return {"owner": filename, "summary": "No API key"}
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "Extract name and summary."},
                {"role": "user", "content": f"Resume:\n{text[:3000]}\n\nReturn:\nNAME: [name]\nSUMMARY: [2-3 sentences]"}
            ],
            temperature=0.2,
            max_tokens=200
        )
        name, summary = filename, ""
        for line in response.choices[0].message.content.split('\n'):
            if line.startswith('NAME:'):
                name = line.replace('NAME:', '').strip()
            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
        return {"owner": name, "summary": summary}
    except:
        return {"owner": filename, "summary": "Error"}

def process_resume(uploaded_file, index, skip_duplicate_check=False):
    """Process resume with duplicate detection"""
    uploaded_file.seek(0, 2)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    
    # Compute file hash BEFORE reading
    file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
    
    reader = PdfReader(uploaded_file)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    cleaned = clean_text(text)
    
    # Compute content fingerprint
    content_fingerprint = DuplicateDetector.compute_content_fingerprint(cleaned)
    
    metadata = generate_metadata(cleaned, uploaded_file.name)
    metadata['num_pages'] = len(reader.pages)
    
    # Generate stable candidate ID
    candidate_id = hashlib.sha256(
        f"{metadata['owner']}_{cleaned[:500]}".encode()
    ).hexdigest()[:12]
    
    return {
        'candidate_id': candidate_id,
        'name': uploaded_file.name,
        'text': cleaned,
        'chunks': chunk_text(cleaned, 5),
        'metadata': metadata,
        'work_history': extract_work_history(cleaned),
        'index': index,
        'file_size_mb': size_mb,
        'file_hash': file_hash,
        'content_fingerprint': content_fingerprint
    }

def check_and_process_resume(uploaded_file, index, existing_resumes):
    """
    Check for duplicates before processing.
    Returns: (resume_data, is_duplicate, duplicate_info)
    """
    # Quick hash check first (no full processing needed)
    file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
    
    for resume in existing_resumes:
        if resume.get('file_hash') == file_hash:
            return None, True, {
                'type': 'exact_file',
                'existing_candidate': resume['metadata']['owner'],
                'message': f"This exact file was already uploaded for {resume['metadata']['owner']}"
            }
    
    # Need to extract text for deeper checks
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    cleaned = clean_text(text)
    
    # Content fingerprint check
    content_fp = DuplicateDetector.compute_content_fingerprint(cleaned)
    for resume in existing_resumes:
        if resume.get('content_fingerprint') == content_fp:
            return None, True, {
                'type': 'content_match',
                'existing_candidate': resume['metadata']['owner'],
                'message': f"This resume has identical content to {resume['metadata']['owner']}'s resume"
            }
    
    # Extract name for name-based checks
    metadata = generate_metadata(cleaned, uploaded_file.name)
    candidate_name = metadata['owner']
    
    # Name checks
    is_dup, dup_type, existing, msg = DuplicateDetector.check_duplicate(
        uploaded_file, cleaned, candidate_name, existing_resumes
    )
    
    if is_dup and dup_type in ['name_match', 'name_similar']:
        # Return partial data for user decision
        return {
            'candidate_name': candidate_name,
            'file_hash': file_hash,
            'content_fingerprint': content_fp,
            'text': cleaned,
            'metadata': metadata
        }, True, {
            'type': dup_type,
            'existing_candidate': existing,
            'message': msg
        }
    
    # No duplicate - process fully
    uploaded_file.seek(0)
    resume_data = process_resume(uploaded_file, index)
    return resume_data, False, None


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def answer_list_all(resumes):
    """Returns ALL candidates"""
    if not resumes:
        return "No resumes uploaded."
    
    lines = [f"I have **{len(resumes)} candidate(s)** loaded:\n"]
    for i, r in enumerate(resumes, 1):
        owner = r['metadata']['owner']
        summary = r['metadata'].get('summary', 'No summary')
        entries = len(r.get('work_history', []))
        lines.append(f"**{i}. {owner}**")
        lines.append(f"   _{summary}_")
        lines.append(f"   📄 {entries} timeline entries\n")
    
    return "\n".join(lines)

def answer_iterate_all(question, resumes, time_period_str):
    """Iterates through EVERY candidate"""
    if not client:
        return "API key required"
    
    query_date = extract_query_date(time_period_str or question)
    date_display = format_date_tuple(query_date) if query_date and query_date[1] else f"year {query_date[0]}" if query_date else "specified period"
    
    results = []
    
    for resume in resumes:
        owner = resume['metadata']['owner']
        
        if query_date:
            entries = filter_entries_by_date(resume.get('work_history', []), query_date)
            if entries:
                entry_text = "\n".join([
                    f"  • {e.get('title', 'Role')} at {e.get('organization', 'Org')} ({e.get('start_display')} - {e.get('end_display')})"
                    for e in entries
                ])
                results.append(f"**{owner}** in {date_display}:\n{entry_text}")
            else:
                results.append(f"**{owner}**: No entries found for {date_display}")
        else:
            entries = resume.get('work_history', [])[:2]
            if entries:
                entry_text = "\n".join([
                    f"  • {e.get('title', 'Role')} at {e.get('organization', 'Org')}"
                    for e in entries
                ])
                results.append(f"**{owner}**:\n{entry_text}")
    
    header = f"Here's what each of the **{len(resumes)} candidates** was doing"
    if query_date:
        header += f" in {date_display}"
    header += ":\n\n"
    
    return header + "\n\n".join(results)

def answer_specific_person(question, person_name, resumes):
    """Answer about a SPECIFIC person only"""
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
    
    prompt = f"""Question: "{question}"

RESUME OF {owner.upper()} (use ONLY this information):
{target['text']}

Work History:
{json.dumps(target.get('work_history', []), indent=2)[:2000]}

Answer using ONLY {owner}'s resume. Never mention other candidates."""

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": f"Answer questions about {owner} ONLY. Never include information from other resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def answer_timeline(question, person_name, resumes, time_period_str):
    """Answer timeline question for a specific person"""
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
    
    query_date = extract_query_date(time_period_str or question)
    if not query_date:
        return "Please specify a time period like 'June 2024' or '2023'."
    
    date_display = format_date_tuple(query_date) if query_date[1] else f"year {query_date[0]}"
    entries = filter_entries_by_date(target.get('work_history', []), query_date)
    
    if entries:
        entries_text = "✅ MATCHING ENTRIES:\n" + "\n".join([
            f"• {e.get('title')} at {e.get('organization')} ({e.get('start_display')} - {e.get('end_display')})"
            for e in entries
        ])
    else:
        entries_text = f"❌ No entries found for {date_display}"
    
    prompt = f"""Question: "{question}"
Person: {owner}
Time Period: {date_display}

{entries_text}

Full Resume:
{target['text'][:3000]}

Answer about {owner} during {date_display}. Use ONLY the matching entries above."""

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": f"Answer about {owner} only. Trust the pre-filtered entries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def answer_general_search(question, resumes):
    """Search ALL resumes for matching criteria"""
    if not client:
        return "API key required"
    
    all_text = []
    for r in resumes:
        owner = r['metadata']['owner']
        all_text.append(f"=== {owner.upper()} ===\n{r['text'][:3000]}")
    
    names = [r['metadata']['owner'] for r in resumes]
    
    prompt = f"""Question: "{question}"

I have {len(resumes)} candidates. Here are ALL their resumes:

{"".join(all_text)}

INSTRUCTIONS:
- Check EVERY candidate: {', '.join(names)}
- List ALL who match the criteria
- Provide evidence from each matching resume
- If none match, say "None of the {len(resumes)} candidates have [criteria]"

You MUST check all {len(resumes)} candidates."""

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": f"Search all {len(resumes)} resumes. Check every candidate. Never skip anyone."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_suggestions(question, answer, target, resumes):
    if not client or not resumes:
        return []
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Generate 3 follow-up questions."},
                {"role": "user", "content": f"Q: {question}\nA: {answer[:200]}\nFocus: {target}\n\n1.\n2.\n3."}
            ],
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
# MAIN HANDLER - WITH ENHANCED ENTITY TRACKING
# =============================================================================

def handle_question(question):
    resumes = st.session_state.resumes
    current_focus = st.session_state.current_focus_person
    last_mentioned = st.session_state.last_mentioned_person
    last_response_entity = st.session_state.get('last_response_entity')
    available = [r['metadata']['owner'] for r in resumes]
    
    # Classify intent with enhanced entity resolution
    classification = classify_intent(
        question, available, current_focus, last_mentioned, last_response_entity
    )
    intent = classification.get('intent')
    target = classification.get('target_person')
    time_period = classification.get('time_period')
    requires_iter = classification.get('requires_iteration', False)
    
    # Update tracking from question if target was explicitly mentioned
    if target and classification.get('resolution_method') == 'explicit_mention':
        EntityTracker.update_from_question(target)
    
    # Route to handler
    if intent == 'list_all':
        answer = answer_list_all(resumes)
        new_focus = None
    elif intent == 'iterate_all' or requires_iter:
        answer = answer_iterate_all(question, resumes, time_period)
        new_focus = None
    elif intent == 'timeline' and target:
        answer = answer_timeline(question, target, resumes, time_period)
        new_focus = target
    elif intent == 'specific_person' and target:
        answer = answer_specific_person(question, target, resumes)
        new_focus = target
    else:
        answer = answer_general_search(question, resumes)
        new_focus = None
    
    # CRITICAL: Update tracking from response content
    response_entity = EntityTracker.update_from_response(answer, available, intent)
    
    # Add response entity to classification for debugging
    classification['response_entity'] = response_entity
    classification['state_after'] = {
        'current_focus': st.session_state.current_focus_person,
        'last_mentioned': st.session_state.last_mentioned_person,
        'last_response_entity': st.session_state.get('last_response_entity')
    }
    
    # Clear focus only for explicit "all" queries
    if intent in ['list_all', 'iterate_all']:
        st.session_state.current_focus_person = None
    
    # Store conversation context
    st.session_state.conversation_context.append({
        'question': question,
        'intent': intent,
        'target': target,
        'resolution': classification.get('resolution_method'),
        'response_entity': response_entity
    })
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context.pop(0)
    
    st.session_state.suggested_questions = generate_suggestions(question, answer, target, resumes)
    
    return answer, classification

# =============================================================================
# UI
# =============================================================================

if not st.session_state.processed:
    st.subheader("📤 Upload Resumes")
    st.info(f"Max file size: {MAX_FILE_SIZE_MB} MB | Duplicates will be detected and rejected")
    
    files = st.file_uploader("Choose PDFs", type=['pdf'], accept_multiple_files=True)
    
    if files:
        valid_files = []
        for f in files:
            is_valid, size_mb, error = check_file_size(f)
            if is_valid:
                valid_files.append(f)
            else:
                st.error(f"❌ {f.name}: {error}")
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} file(s) ready for processing")
            
            if st.button("🚀 Process Resumes", type="primary"):
                if not OPENAI_API_KEY:
                    st.error("Enter API key first")
                else:
                    progress = st.progress(0)
                    processed = []
                    duplicates_found = []
                    
                    for i, f in enumerate(valid_files):
                        st.text(f"Processing: {f.name}")
                        
                        # Check for duplicates against already processed in this batch
                        resume_data, is_dup, dup_info = check_and_process_resume(f, i, processed)
                        
                        if is_dup:
                            dup_type = dup_info['type']
                            if dup_type in ['exact_file', 'content_match']:
                                st.warning(f"⚠️ Skipped {f.name}: {dup_info['message']}")
                                duplicates_found.append({
                                    'file': f.name,
                                    'reason': dup_info['message']
                                })
                            elif dup_type in ['name_match', 'name_similar']:
                                st.warning(f"⚠️ Skipped {f.name}: {dup_info['message']}")
                                duplicates_found.append({
                                    'file': f.name,
                                    'reason': dup_info['message']
                                })
                        else:
                            processed.append(resume_data)
                        
                        progress.progress((i+1) / len(valid_files))
                    
                    if processed:
                        vector_db, chunks_meta, model = build_vector_db(processed)
                        st.session_state.resumes = processed
                        st.session_state.vector_db = vector_db
                        st.session_state.chunks_metadata = chunks_meta
                        st.session_state.embedding_model = model
                        st.session_state.processed = True
                        st.session_state.duplicate_warnings = duplicates_found
                        
                        msg = f"✅ Processed {len(processed)} unique resume(s)!"
                        if duplicates_found:
                            msg += f" ({len(duplicates_found)} duplicate(s) skipped)"
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error("All files were duplicates. No resumes processed.")

if st.session_state.processed:
    st.subheader(f"📚 {len(st.session_state.resumes)} Unique Candidates Loaded")
    
    # Show any duplicate warnings from upload
    if st.session_state.get('duplicate_warnings'):
        with st.expander("⚠️ Duplicates Detected During Upload"):
            for dup in st.session_state.duplicate_warnings:
                st.caption(f"• {dup['file']}: {dup['reason']}")
    
    cols = st.columns(min(len(st.session_state.resumes), 4))
    for i, r in enumerate(st.session_state.resumes):
        with cols[i % 4]:
            with st.container(border=True):
                st.write(f"**{r['metadata']['owner']}**")
                st.caption(f"ID: {r.get('candidate_id', 'N/A')[:8]}...")
                st.caption(f"{len(r.get('work_history', []))} entries")
    
    # RESET BUTTON
    col1, col2 = st.columns([1, 4])
    if col1.button("🗑️ Reset All", type="secondary"):
        perform_complete_reset()
        st.rerun()
    
    with st.expander("📤 Add More Resumes"):
        new_files = st.file_uploader("More PDFs", type=['pdf'], accept_multiple_files=True, key="add")
        if new_files:
            valid = [f for f in new_files if check_file_size(f)[0]]
            if valid and st.button("➕ Add"):
                start = len(st.session_state.resumes)
                added = 0
                for i, f in enumerate(valid):
                    resume_data, is_dup, dup_info = check_and_process_resume(
                        f, start + i, st.session_state.resumes
                    )
                    if is_dup:
                        st.warning(f"⚠️ Skipped {f.name}: {dup_info['message']}")
                    else:
                        st.session_state.resumes.append(resume_data)
                        added += 1
                
                if added > 0:
                    vector_db, chunks_meta, model = build_vector_db(st.session_state.resumes)
                    st.session_state.vector_db = vector_db
                    st.session_state.chunks_metadata = chunks_meta
                    st.session_state.embedding_model = model
                    st.success(f"Added {added} new resume(s)")
                st.rerun()
    
    st.divider()
    st.subheader("💬 Chat")
    
    # Show current tracking state with more detail
    tracking_info = []
    if st.session_state.current_focus_person:
        tracking_info.append(f"**Focused on:** {st.session_state.current_focus_person}")
    if st.session_state.get('last_response_entity'):
        tracking_info.append(f"**Last mentioned in answer:** {st.session_state.last_response_entity}")
    
    if tracking_info:
        st.info(" | ".join(tracking_info) + "\n\n_Pronouns (he/she/they) will refer to the last mentioned person_")
    else:
        st.info("📍 No specific focus - ask about anyone")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "debug" in msg and msg["debug"]:
                with st.expander("🔍 Debug Info"):
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
        question = st.chat_input("Ask about candidates...")
    
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, classification = handle_question(question)
                st.markdown(answer)
                with st.expander("🔍 Debug Info"):
                    st.json(classification)
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer, 
            "debug": classification
        })
        st.rerun()