from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

# =============================================================================
# FILE TYPE CLASSIFICATION SYSTEM
# =============================================================================

class FileType(Enum):
    RESUME = "resume"
    NON_RESUME = "non_resume"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    file_type: FileType
    confidence: float
    justification: str
    detected_document_type: str  # More specific: "cover_letter", "transcript", etc.

@dataclass
class InjectionReport:
    suspicious_patterns_found: bool
    zero_width_chars_removed: int
    whitespace_anomalies: int
    suspicious_phrases: List[str]
    removed_segments: List[str]
    risk_score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
