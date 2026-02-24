import hashlib
from typing import List, Dict, Tuple
from pypdf import PdfReader
from openai import OpenAI

from modules.models import FileType, InjectionReport, ClassificationResult
from modules.injection_guard import PromptInjectionGuard
from modules.classifier import FileTypeClassifier
from modules.file_utils import DuplicateDetector
from modules.text_processing import clean_text, chunk_text
from modules.extraction import extract_work_history_guarded, extract_non_resume_data_guarded, generate_metadata_guarded

# =============================================================================
# DOCUMENT PROCESSING WITH CLASSIFICATION & SANITIZATION
# =============================================================================

def process_document_full(uploaded_file, index: int, client: OpenAI) -> Tuple[Dict, InjectionReport, ClassificationResult]:
    """
    Full document processing pipeline:
    1. Extract raw text
    2. Sanitize for injection
    3. Classify document type
    4. Extract structured data based on type
    """
    # Get file hash
    file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
    
    # Extract raw text from PDF
    reader = PdfReader(uploaded_file)
    raw_text = "".join([p.extract_text() or "" for p in reader.pages])
    
    # STAGE 2: Sanitize for prompt injection
    sanitized_text, injection_report = PromptInjectionGuard.sanitize_document(raw_text)
    
    # Basic cleaning
    cleaned_text = clean_text(sanitized_text)
    
    # Compute content fingerprint on sanitized text
    content_fingerprint = DuplicateDetector.compute_content_fingerprint(cleaned_text)
    
    # STAGE 3: Classify document type
    classification = FileTypeClassifier.classify(cleaned_text, client)
    
    # STAGE 4: Extract based on document type
    if classification.file_type == FileType.RESUME:
        # Resume extraction
        metadata = generate_metadata_guarded(cleaned_text, uploaded_file.name, client)
        metadata['num_pages'] = len(reader.pages)
        work_history = extract_work_history_guarded(cleaned_text, client)
        
        doc_id = hashlib.sha256(
            f"{metadata['owner']}_{cleaned_text[:500]}".encode()
        ).hexdigest()[:12]
        
        doc_data = {
            'doc_id': doc_id,
            'file_type': 'resume',
            'name': uploaded_file.name,
            'text': cleaned_text,
            'chunks': chunk_text(cleaned_text, 5),
            'metadata': metadata,
            'work_history': work_history,
            'index': index,
            'file_hash': file_hash,
            'content_fingerprint': content_fingerprint,
            'classification': {
                'type': classification.detected_document_type,
                'confidence': classification.confidence,
                'justification': classification.justification
            },
            'injection_report': {
                'risk_score': injection_report.risk_score,
                'suspicious': injection_report.suspicious_patterns_found,
                'zero_width_removed': injection_report.zero_width_chars_removed
            }
        }
    else:
        # Non-resume extraction
        extracted_data = extract_non_resume_data_guarded(
            cleaned_text, 
            classification.detected_document_type, 
            client
        )
        
        doc_id = hashlib.sha256(
            f"{uploaded_file.name}_{cleaned_text[:500]}".encode()
        ).hexdigest()[:12]
        
        doc_data = {
            'doc_id': doc_id,
            'file_type': 'non_resume',
            'detected_type': classification.detected_document_type,
            'name': uploaded_file.name,
            'text': cleaned_text,
            'chunks': chunk_text(cleaned_text, 5),
            'metadata': {
                'title': extracted_data.get('title', uploaded_file.name),
                'author': extracted_data.get('author'),
                'date': extracted_data.get('date'),
                'summary': extracted_data.get('summary', ''),
                'num_pages': len(reader.pages)
            },
            'extracted_data': extracted_data,
            'index': index,
            'file_hash': file_hash,
            'content_fingerprint': content_fingerprint,
            'classification': {
                'type': classification.detected_document_type,
                'confidence': classification.confidence,
                'justification': classification.justification
            },
            'injection_report': {
                'risk_score': injection_report.risk_score,
                'suspicious': injection_report.suspicious_patterns_found,
                'zero_width_removed': injection_report.zero_width_chars_removed
            }
        }
    
    return doc_data, injection_report, classification


def check_and_process_document(uploaded_file, index: int, existing_docs: List[Dict], client: OpenAI):
    """
    Check for duplicates, then process document with full pipeline.
    Returns: (doc_data, is_duplicate, duplicate_info, injection_report, classification)
    """
    # Quick hash check first
    file_hash = DuplicateDetector.compute_file_hash(uploaded_file)
    
    for doc in existing_docs:
        if doc.get('file_hash') == file_hash:
            return None, True, {
                'type': 'exact_file',
                'existing': doc['metadata'].get('owner', doc['metadata'].get('title', 'Unknown')),
                'message': "This exact file was already uploaded"
            }, None, None
    
    # Process fully
    uploaded_file.seek(0)
    doc_data, injection_report, classification = process_document_full(uploaded_file, index, client)
    
    # Check content fingerprint
    for doc in existing_docs:
        if doc.get('content_fingerprint') == doc_data.get('content_fingerprint'):
            return None, True, {
                'type': 'content_match',
                'existing': doc['metadata'].get('owner', doc['metadata'].get('title', 'Unknown')),
                'message': "This document has identical content"
            }, injection_report, classification
    
    # Name check for resumes
    if doc_data.get('file_type') == 'resume':
        new_name = doc_data['metadata'].get('owner', '')
        if new_name:
            is_dup, dup_type, existing, msg = DuplicateDetector.check_duplicate(
                uploaded_file, doc_data['text'], new_name, existing_docs
            )
            if is_dup:
                return None, True, {
                    'type': dup_type,
                    'existing': existing,
                    'message': msg
                }, injection_report, classification
    
    return doc_data, False, None, injection_report, classification
