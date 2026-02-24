import streamlit as st
from modules.memory import init_short_term_memory, init_user_facts_memory

# =============================================================================
# SESSION STATE MANAGEMENT  (single-tenant)
# =============================================================================

def get_default_state():
    return {
        'documents': [],
        'processed': False,
        'chat_history': [],
        'conversation_context': [],
        'short_term_memory': init_short_term_memory(),
        'user_facts_memory': init_user_facts_memory(),
        'suggested_questions': [],
        'vector_db': None,
        'chunks_metadata': [],
        'embedding_model': None,
        'pending_question': None,
        'shutdown_requested': False,
        'file_rejected': False,
        'reset_counter': 0,
        'injection_alerts': [],
    }

def perform_complete_reset():
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
