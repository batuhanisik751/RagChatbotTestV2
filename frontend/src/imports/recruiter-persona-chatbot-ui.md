Design a new UI/UX for my existing app. Do not change the core functionality. Your job is to understand what the app already does and redesign the interface so it is clearer, more polished, and easier to use.

Do NOT use screenshots. Base the design only on the product behavior and requirements described below.

This is a recruiter-facing, persona-based RAG chatbot. The chatbot answers questions as a specific candidate (in first person), using uploaded documents as ground truth (resume, cover letters, transcripts, etc.). It is not a generic assistant and not a multi-user platform. It is a single-tenant persona experience.

Please design the UI around the following actual product behavior.

PRODUCT IDEA (what the app does)
This app lets me upload PDF documents about one candidate, processes them through a secure document pipeline, builds semantic search over them, and then allows a recruiter to chat with an AI agent that answers as the candidate. The AI can also use tools (web search, GitHub, LinkedIn, weather) when needed. The answer is grounded in uploaded documents + persona config + memory.

The AI should feel like “chatting with the candidate,” not “chatting with an assistant about the candidate.”

CORE EXPERIENCE FLOW
1. User lands on a home page and can navigate to the chatbot experience.
2. App loads a single persona from a config file (name, role, location, tone, speaking style, bio, contact links).
3. User uploads one or more PDF documents.
4. Each file is validated and processed through a pipeline:
   - file size validation
   - duplicate detection
   - prompt injection sanitization/risk scoring
   - document type classification (resume vs non-resume)
   - structured extraction (work/education or key facts/entities)
   - chunking + semantic index (FAISS)
5. After processing, the app shows loaded documents, document types, and security risk indicators.
6. User chats with the AI agent.
7. The agent may call tools (semantic search, web, GitHub, LinkedIn, weather), then synthesizes a first-person answer.
8. Follow-up question suggestions appear after answers.
9. Optional debug details show tool traces and agent rounds.

IMPORTANT FUNCTIONAL CONSTRAINTS
- Single-tenant persona (one candidate identity per deployment).
- PDF-only uploads.
- Max file size per file is 2 MB.
- Documents are processed in-session (ephemeral session state).
- Semantic search is over uploaded document chunks.
- The app has a security posture: suspicious/injection-like content is detected and risk-scored.
- The app exposes debug info per assistant response (tool calls, rounds, errors).
- There is a “Reset All” action and a “Quit app” action.
- There is an “Add More Documents” flow after initial processing.

CURRENT USER-VISIBLE FEATURES (must be represented in redesigned UI)
1. Persona / Configuration status
- Persona loaded status (success or error if missing/invalid).
- Persona name and role shown.
- API key status shown (present or missing).
- Model names shown (classifier / extraction / answer).
- Available agent tools shown (web search, semantic search, weather, GitHub, LinkedIn).

2. Document upload and processing
- Multi-file PDF uploader.
- Per-file validation error (size limit).
- “Process Documents” primary action.
- Processing progress bar and current file status text.
- Duplicate file warnings (exact file / identical content / same candidate name).
- Classification result feedback for each processed file (resume or non-resume subtype + confidence).
- Security warning when injection risk score is elevated.
- Expandable details for suspicious phrases during processing (at least conceptually supported).

3. Processed documents overview
- Count of loaded documents.
- Document cards/grid with:
  - document display name
  - type (resume / transcript / cover letter / etc.)
  - security risk indicator (green/yellow/red + percentage)
- Global warning if one or more docs have elevated risk scores.
- “Reset All” action.
- “Add More Documents” expandable flow with uploader and add action.

4. Chat interface
- Chat history (user + assistant messages).
- AI speaks as the persona in first person.
- Suggestions (3 follow-up recruiter questions) shown as clickable chips/buttons.
- Chat input for recruiter questions.
- Assistant loading state (“Thinking…”).
- Expandable debug panel for assistant messages showing tool-call trace and agent metadata.

5. Sidebar / support panel (or equivalent layout)
- Configuration and system status.
- Loaded documents summary (with risk icons).
- Security alerts list (recent alerts).
- Quit app confirmation flow.

AI AGENT BEHAVIOR (important for UI semantics)
The app uses an agent loop (up to 6 tool-call rounds per message). The agent decides whether to call:
- semantic_search (candidate documents, primary ground truth)
- web_search (external info)
- github_search (candidate repos/profile)
- linkedin_search (candidate professional profile info)
- weather_lookup (small talk/logistics)

The user does not manually trigger tools; tools are automatic. The UI should communicate that the assistant may “research” before replying, without exposing raw JSON by default.

MEMORY / CONTEXT BEHAVIOR (inform UI copy and interaction)
- Short-term conversation memory keeps recent turns + summarized older turns.
- Long-term “user facts” memory stores profile info, roles, organizations, skills, key facts from persona/documents.
- This memory supports conversational continuity.
- The user does not directly edit memory in the current UI, but the system uses it internally.

SECURITY / TRUST BEHAVIOR (very important for design)
The app is security-conscious and scans documents for prompt injection attempts and suspicious formatting/unicode tricks.
Each document gets a risk score (0.0–1.0), with visual severity:
- Low (green)
- Medium (yellow)
- High (red)

Design the UI so the app feels trustworthy and transparent:
- Show security status clearly but not alarmist.
- Make warnings informative.
- Preserve access to details/debug for advanced users.

DATA TYPES THE UI SHOULD ASSUME
Document records may include:
- name/title/owner display name
- file type (resume / non_resume)
- detected type (cover_letter, transcript, report, etc.)
- metadata summary
- extracted work/education history (for resumes)
- injection risk score
- classification confidence

Chat debug info may include:
- agent_rounds
- tool_calls (tool name, args, result count, error)
- finish_reason
- error states

DESIGN GOALS (what I want from the redesign)
- Make the app feel like a polished recruiter interview/chat experience.
- Improve clarity of the upload-to-chat transition.
- Make document processing and security status easier to understand at a glance.
- Keep advanced debug transparency available without cluttering the main experience.
- Improve information architecture and hierarchy (currently it is a Streamlit-style utility layout).
- Design for desktop first, but include mobile-responsive behavior.
- Make the persona identity feel central (who the recruiter is chatting with).
- Emphasize trust, evidence, and grounding in uploaded documents.

WHAT TO DELIVER (Figma output expectations)
Please generate:
1. A clear information architecture for the app.
2. A redesigned desktop UI (high fidelity) for key screens/states.
3. Mobile-responsive variants for the main screens.
4. A component system / design tokens (buttons, chips, cards, alerts, badges, chat bubbles, status panels, progress states, debug panels).
5. State variants for empty/loading/error/success/warning conditions.
6. Interaction notes for major flows (upload, processing, chat, suggestions, debug expanders, add docs, reset).
7. Suggested microcopy improvements for trust/security and recruiter-friendly messaging.

KEY SCREENS / STATES TO DESIGN
- Home page / landing page with navigation entry into the chatbot.
- Empty state: app opened, persona status visible, no docs uploaded yet.
- No API key warning state.
- Persona missing/error state.
- Files selected state (ready to process).
- Processing in progress state (progress + per-file statuses).
- Duplicate file warning state.
- Security risk warning state (with details affordance).
- Documents loaded dashboard state.
- Add more documents state.
- Chat state with history.
- Chat thinking/loading state.
- Chat with suggestion chips.
- Chat with debug panel expanded.
- Error states (tool/network/model failure, no docs processed).

VISUAL DIRECTION
Choose a professional, trust-first visual style suitable for recruiter conversations and candidate presentation.
Avoid generic “toy chatbot” UI patterns.
The UI should communicate:
- credibility
- grounded answers
- security awareness
- clear workflow progression (home → chatbot → upload → process → chat)

Please prioritize usability, hierarchy, and state clarity over decorative UI.