from modules.date_utils import extract_query_date

# =============================================================================
# QUERY ANALYSIS  (single-tenant — no multi-candidate routing)
# =============================================================================


def analyse_query(question: str) -> dict:
    """
    Lightweight query analysis for single-tenant persona.

    Returns a dict with:
        time_period  – extracted (year, month|None) tuple or None
    """
    time_period = extract_query_date(question)
    return {"time_period": time_period}
