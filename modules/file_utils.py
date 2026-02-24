import re
import hashlib

from modules.config import MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES

# =============================================================================
# FILE SIZE GUARDRAIL
# =============================================================================

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
# DUPLICATE DETECTION SYSTEM
# =============================================================================

class DuplicateDetector:
    @staticmethod
    def compute_file_hash(uploaded_file):
        uploaded_file.seek(0)
        file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
        uploaded_file.seek(0)
        return file_hash
    
    @staticmethod
    def compute_content_fingerprint(text):
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = ' '.join(normalized.split())
        words = sorted(set(normalized.split()))
        fingerprint_base = ' '.join(words[:150])
        return hashlib.sha256(fingerprint_base.encode()).hexdigest()
    
    @staticmethod
    def normalize_name(name):
        name = re.sub(r'\b(mr|mrs|ms|dr|prof|jr|sr|ii|iii|iv)\b\.?', '', name.lower())
        name = re.sub(r'[^\w\s]', '', name)
        return ' '.join(name.split())
    
    @staticmethod
    def name_similarity(name1, name2):
        n1_parts = set(DuplicateDetector.normalize_name(name1).split())
        n2_parts = set(DuplicateDetector.normalize_name(name2).split())
        if not n1_parts or not n2_parts:
            return 0.0
        intersection = len(n1_parts & n2_parts)
        union = len(n1_parts | n2_parts)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def check_duplicate(uploaded_file, extracted_text, candidate_name, existing_docs):
        if not existing_docs:
            return False, None, None, None
        
        new_file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
        for doc in existing_docs:
            if doc.get('file_hash') == new_file_hash:
                existing_name = doc.get('metadata', {}).get('owner') or doc.get('metadata', {}).get('title') or 'Unknown'
                return True, 'exact_file', existing_name, \
                    f"This exact file was already uploaded"
        
        new_fingerprint = DuplicateDetector.compute_content_fingerprint(extracted_text)
        for doc in existing_docs:
            if doc.get('content_fingerprint') == new_fingerprint:
                existing_name = doc.get('metadata', {}).get('owner') or doc.get('metadata', {}).get('title') or 'Unknown'
                return True, 'content_match', existing_name, \
                    f"This document has identical content"
        
        if candidate_name:
            new_name_norm = DuplicateDetector.normalize_name(candidate_name)
            for doc in existing_docs:
                existing_name = doc.get('metadata', {}).get('owner') or ''
                if existing_name:
                    existing_name_norm = DuplicateDetector.normalize_name(existing_name)
                    if new_name_norm == existing_name_norm:
                        return True, 'name_match', existing_name, \
                            f"A document for {existing_name} already exists"
        
        return False, None, None, None
