# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
CHAT_MODEL = "gpt-5.2"
ANSWER_MODEL = "gpt-5.2"
EXTRACTION_MODEL = "gpt-5.2"
CLASSIFIER_MODEL = "gpt-5.2"

# =============================================================================
# FILE SIZE GUARDRAIL
# =============================================================================
MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# =============================================================================
# PERSONA (single-tenant identity)
# =============================================================================
import os as _os
PERSONA_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "persona.yaml",
)
