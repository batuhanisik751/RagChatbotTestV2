import re
import json
from typing import List, Dict
from openai import OpenAI

from modules.config import EXTRACTION_MODEL
from modules.prompts import GuardedPrompts
from modules.date_utils import parse_date_to_yyyymm, format_date_tuple

# =============================================================================
# DOCUMENT PROCESSING WITH CLASSIFICATION & SANITIZATION
# =============================================================================

def extract_work_history_guarded(text: str, client: OpenAI) -> List[Dict]:
    """Extract work history with guardrail prompts."""
    if not client:
        return []
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": GuardedPrompts.RESUME_EXTRACTION_SYSTEM},
                {"role": "user", "content": GuardedPrompts.RESUME_EXTRACTION_USER.format(
                    document_text=text[:5000]
                )}
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

def extract_non_resume_data_guarded(text: str, document_type: str, client: OpenAI) -> Dict:
    """Extract structured data from non-resume documents."""
    if not client:
        return {}
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": GuardedPrompts.NON_RESUME_EXTRACTION_SYSTEM},
                {"role": "user", "content": GuardedPrompts.NON_RESUME_EXTRACTION_USER.format(
                    document_type=document_type,
                    document_text=text[:5000]
                )}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = re.sub(r'^```json?\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
        return json.loads(result)
    except:
        return {}

def generate_metadata_guarded(text: str, filename: str, client: OpenAI) -> Dict:
    """Generate metadata with guardrail instructions."""
    if not client:
        return {"owner": filename, "summary": "No API key"}
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": GuardedPrompts.GUARDRAIL_PREAMBLE + 
                 "Extract the person's name and a brief factual summary from this resume. "
                 "Do not include self-promotional claims as facts."},
                {"role": "user", "content": f"Resume text:\n{text[:3000]}\n\nReturn:\nNAME: [full name]\nSUMMARY: [2-3 factual sentences about experience]"}
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
