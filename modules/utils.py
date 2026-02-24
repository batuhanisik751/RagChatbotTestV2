from typing import Dict


def get_document_display_name(doc: Dict) -> str:
    """Safely get display name for a document."""
    if doc.get('file_type') == 'resume':
        name = doc.get('metadata', {}).get('owner')
    else:
        name = doc.get('metadata', {}).get('title')
    
    if not name:
        name = doc.get('metadata', {}).get('owner') or doc.get('metadata', {}).get('title') or doc.get('name', 'Unknown')
    
    return name or 'Unknown'
