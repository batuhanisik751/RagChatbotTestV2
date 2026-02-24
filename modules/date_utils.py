import re
from datetime import datetime
from dateutil import parser as date_parser

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
            sv = start[0] * 100 + start[1] if start else 0
            ev = end[0] * 100 + end[1] if end else 999912
            qv = query_year * 100 + query_month
            if sv <= qv <= ev:
                matching.append(entry)
        else:
            sv = start[0] * 100 + start[1] if start else 0
            ev = end[0] * 100 + end[1] if end else 999912
            ysv, yev = query_year * 100 + 1, query_year * 100 + 12
            if not (ev < ysv or sv > yev):
                matching.append(entry)
    return matching
