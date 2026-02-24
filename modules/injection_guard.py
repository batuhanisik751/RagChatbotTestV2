import re
import unicodedata
from typing import List, Tuple

from modules.models import InjectionReport

# =============================================================================
# PROMPT INJECTION DETECTION & SANITIZATION
# =============================================================================

class PromptInjectionGuard:
    """
    Multi-layer defense against prompt injection in documents.
    """
    
    # Zero-width and invisible characters
    ZERO_WIDTH_CHARS = {
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\ufeff',  # Zero-width no-break space (BOM)
        '\u180e',  # Mongolian vowel separator
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u206a',  # Inhibit symmetric swapping
        '\u206b',  # Activate symmetric swapping
        '\u206c',  # Inhibit Arabic form shaping
        '\u206d',  # Activate Arabic form shaping
        '\u206e',  # National digit shapes
        '\u206f',  # Nominal digit shapes
    }
    
    # Suspicious instruction patterns that might be injected
    INJECTION_PATTERNS = [
        # Direct instruction attempts
        r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)',
        r'disregard\s+(all\s+)?(previous|prior|above)',
        r'forget\s+(everything|all|what)\s+(you|i)\s+(said|told|know)',
        r'new\s+instructions?\s*[:=]',
        r'system\s*[:=]\s*you\s+are',
        r'<\s*system\s*>',
        r'\[\s*system\s*\]',
        
        # Ranking/recommendation manipulation
        r'(this|the)\s+(candidate|person|applicant)\s+(is|should\s+be)\s+(the\s+)?(best|top|first|ideal|perfect)',
        r'rank\s+(this|me|them)\s+(first|highest|top)',
        r'(always|must|should)\s+(recommend|select|choose|pick)\s+(this|me)',
        r'(hire|select|choose)\s+(this|me)\s+(immediately|first|now)',
        r'(perfect|ideal|best)\s+(candidate|fit|match)\s+for\s+(any|all|every)',
        
        # Hidden endorsements
        r'secretly\s+(note|remember|know)',
        r'hidden\s+(message|instruction|note)',
        r'(note|remember)\s*:\s*(this|the)\s+(candidate|person)',
        
        # Role manipulation
        r'you\s+are\s+(now|actually)',
        r'pretend\s+(to\s+be|you\s+are)',
        r'act\s+as\s+(if|though)',
        r'roleplay\s+as',
        
        # Output manipulation
        r'(always|must|should)\s+(say|respond|answer|output)',
        r'your\s+(response|answer|output)\s+(must|should|will)\s+be',
        r'respond\s+with\s+only',
    ]
    
    # Compile patterns for efficiency
    COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
    
    @classmethod
    def detect_zero_width_chars(cls, text: str) -> Tuple[str, int, List[str]]:
        """Remove zero-width characters and return cleaned text with count."""
        removed = []
        count = 0
        cleaned = []
        
        for char in text:
            if char in cls.ZERO_WIDTH_CHARS:
                count += 1
                if len(removed) < 10:  # Limit stored examples
                    removed.append(f"U+{ord(char):04X}")
            else:
                cleaned.append(char)
        
        return ''.join(cleaned), count, removed
    
    @classmethod
    def detect_whitespace_encoding(cls, text: str) -> Tuple[str, int, List[str]]:
        """
        Detect text potentially encoded in whitespace patterns.
        Checks for unusual whitespace sequences that might encode data.
        """
        anomalies = 0
        suspicious_segments = []
        
        # Pattern: sequences of tabs/spaces that could encode binary
        whitespace_pattern = re.compile(r'([ \t]{20,})')
        matches = whitespace_pattern.findall(text)
        
        for match in matches:
            # Check if whitespace has suspicious pattern (alternating)
            if len(set(match)) > 1:  # Mix of spaces and tabs
                anomalies += 1
                if len(suspicious_segments) < 5:
                    suspicious_segments.append(f"Suspicious whitespace block ({len(match)} chars)")
        
        # Pattern: excessive line breaks with whitespace
        excessive_breaks = re.compile(r'(\n\s*){5,}')
        if excessive_breaks.search(text):
            anomalies += 1
            suspicious_segments.append("Excessive line breaks with whitespace")
        
        # Normalize excessive whitespace
        cleaned = re.sub(r'[ \t]{10,}', ' ', text)
        cleaned = re.sub(r'\n{4,}', '\n\n', cleaned)
        
        return cleaned, anomalies, suspicious_segments
    
    @classmethod
    def detect_injection_phrases(cls, text: str) -> List[Tuple[str, str]]:
        """Detect suspicious instruction-like phrases in document."""
        found = []
        text_lower = text.lower()
        
        for pattern in cls.COMPILED_PATTERNS:
            matches = pattern.finditer(text_lower)
            for match in matches:
                # Get context around match
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].replace('\n', ' ')
                found.append((match.group(), context))
        
        return found
    
    @classmethod
    def detect_unicode_smuggling(cls, text: str) -> Tuple[str, List[str]]:
        """
        Detect and neutralize Unicode smuggling techniques.
        - Homoglyph attacks (lookalike characters)
        - Tag characters (U+E0000 range)
        - Variation selectors used for hiding
        """
        issues = []
        cleaned_chars = []
        
        for char in text:
            code = ord(char)
            
            # Tag characters (U+E0000 - U+E007F) - used for invisible text
            if 0xE0000 <= code <= 0xE007F:
                issues.append(f"Tag character U+{code:04X} removed")
                continue
            
            # Variation selectors (except common ones)
            if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
                # Keep only if following emoji/symbol
                if cleaned_chars and ord(cleaned_chars[-1]) > 0x2000:
                    cleaned_chars.append(char)
                else:
                    issues.append(f"Orphan variation selector removed")
                continue
            
            # Private Use Area characters (sometimes used for hiding)
            if 0xE000 <= code <= 0xF8FF or 0xF0000 <= code <= 0xFFFFD:
                issues.append(f"Private use character U+{code:04X} removed")
                continue
            
            cleaned_chars.append(char)
        
        return ''.join(cleaned_chars), issues
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Apply Unicode normalization to prevent homoglyph attacks."""
        # NFKC normalization converts lookalike characters to standard forms
        return unicodedata.normalize('NFKC', text)
    
    @classmethod
    def sanitize_document(cls, raw_text: str) -> Tuple[str, InjectionReport]:
        """
        Full sanitization pipeline for document text.
        Returns sanitized text and detailed report.
        """
        suspicious_patterns_found = False
        removed_segments = []
        details = {}
        
        # Step 1: Detect and remove zero-width characters
        text, zw_count, zw_removed = cls.detect_zero_width_chars(raw_text)
        if zw_count > 0:
            details['zero_width'] = {'count': zw_count, 'examples': zw_removed}
        
        # Step 2: Detect whitespace encoding
        text, ws_anomalies, ws_suspicious = cls.detect_whitespace_encoding(text)
        if ws_anomalies > 0:
            details['whitespace'] = {'anomalies': ws_anomalies, 'suspicious': ws_suspicious}
            removed_segments.extend(ws_suspicious)
        
        # Step 3: Unicode smuggling detection
        text, unicode_issues = cls.detect_unicode_smuggling(text)
        if unicode_issues:
            details['unicode'] = unicode_issues
            removed_segments.extend(unicode_issues)
        
        # Step 4: Normalize Unicode
        text = cls.normalize_text(text)
        
        # Step 5: Detect injection phrases (don't remove, just flag)
        injection_matches = cls.detect_injection_phrases(text)
        suspicious_phrases = []
        if injection_matches:
            suspicious_patterns_found = True
            suspicious_phrases = [match for match, _ in injection_matches]
            details['injection_phrases'] = [
                {'phrase': match, 'context': ctx} 
                for match, ctx in injection_matches[:10]
            ]
        
        # Calculate risk score
        risk_score = cls.calculate_risk_score(
            zw_count, ws_anomalies, len(unicode_issues), len(injection_matches)
        )
        
        report = InjectionReport(
            suspicious_patterns_found=suspicious_patterns_found,
            zero_width_chars_removed=zw_count,
            whitespace_anomalies=ws_anomalies,
            suspicious_phrases=suspicious_phrases,
            removed_segments=removed_segments,
            risk_score=risk_score,
            details=details
        )
        
        return text, report
    
    @classmethod
    def calculate_risk_score(cls, zw_count: int, ws_anomalies: int, 
                            unicode_issues: int, injection_phrases: int) -> float:
        """Calculate overall injection risk score."""
        score = 0.0
        
        # Zero-width chars are very suspicious
        if zw_count > 0:
            score += min(0.3, zw_count * 0.05)
        
        # Whitespace anomalies
        if ws_anomalies > 0:
            score += min(0.2, ws_anomalies * 0.1)
        
        # Unicode issues
        if unicode_issues > 0:
            score += min(0.2, unicode_issues * 0.05)
        
        # Injection phrases are highest risk
        if injection_phrases > 0:
            score += min(0.5, injection_phrases * 0.2)
        
        return min(1.0, score)
