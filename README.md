# 📝 Document Coach - Intelligent Resume & Document Analysis Chatbot

A RAG (Retrieval-Augmented Generation) powered chatbot that enables natural language queries over multiple resumes and documents. Features advanced security guardrails against prompt injection attacks and intelligent file type classification.

---

## 🎯 Overview

Document Coach is an AI-powered document analysis system designed for hiring workflows and document review. Upload PDF resumes and other documents, then ask natural language questions about candidates. The system automatically classifies document types, extracts structured information, and protects against manipulation attempts embedded in documents.

**Key Differentiators:**
- **Security-First Design** — Multi-layer prompt injection detection and sanitization
- **Universal Document Support** — Handles resumes, cover letters, transcripts, and other documents
- **Intelligent Classification** — Dedicated LLM classifies document types before extraction

---

## ✨ Features

### Core Functionality
- **📄 Multi-Document Upload** — Drag and drop PDF files (max 2MB each)
- **🔍 Semantic Search** — Find candidates based on skills, experience, and qualifications
- **💬 Natural Language Queries** — Ask questions like "Who has Python experience?" or "What was John doing in 2024?"
- **🧠 Conversation Context** — Maintains context across questions with pronoun resolution ("What about his education?")
- **📊 Document Classification** — Automatically identifies resumes vs. cover letters, transcripts, portfolios, etc.

### Advanced RAG Pipeline
- **LLM-based File Classification** — Separate classification step determines document type before extraction
- **Structured Data Extraction** — Extracts work history, education, skills from resumes; key entities and facts from other documents
- **Temporal Query Support** — Handles date-based queries ("What was John doing in December 2024?")
- **Entity Tracking** — Resolves pronouns using conversation history and response context
- **Multi-candidate Queries** — Supports comparison queries across all loaded candidates

### Security & Guardrails
- **🛡️ Prompt Injection Detection** — Detects and neutralizes manipulation attempts in documents
- **🔒 Zero-Width Character Removal** — Strips invisible Unicode characters used for hidden text
- **🧹 Content Sanitization** — Removes whitespace encoding, Unicode smuggling, and homoglyph attacks
- **⚠️ Risk Scoring** — Calculates and displays injection risk scores for each document
- **🚫 Guarded Prompts** — All LLM calls include injection-resistant system instructions
- **📋 Duplicate Detection** — Prevents duplicate uploads via file hash, content fingerprint, and name matching

### Example Questions You Can Ask:
- "Who do you have?" / "List all candidates"
- "Who has Python experience?"
- "Tell me about John's background"
- "What was she doing in January 2024?" (uses conversation context)
- "Compare candidates for a backend role"
- "What skills does Sarah have?"
- "Who worked at Google?"

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web application framework & UI |
| **LLM** | OpenAI GPT-4o / GPT-4o-mini | Classification, extraction, and answer generation |
| **Vector Database** | FAISS | In-memory vector storage for semantic search |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Document and query embeddings |
| **PDF Processing** | pypdf | PDF text extraction |
| **Language** | Python 3.8+ | Core programming language |

### Model Configuration

| Role | Model | Purpose |
|------|-------|---------|
| **Classifier** | `gpt-4o-mini` | Determine document type (resume vs. non-resume) |
| **Extractor** | `gpt-4o-mini` | Extract structured data from documents |
| **Answerer** | `gpt-4o` | Generate final answers to user queries |
| **Chat** | `gpt-4o-mini` | Query suggestions and conversation support |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT UPLOAD PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

User uploads PDF
       │
       ▼
┌──────────────────┐
│ FILE VALIDATION  │ ◄── Size check (2MB limit), duplicate detection
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ RAW EXTRACTION   │ ◄── pypdf extracts text from PDF
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                  SANITIZATION LAYER                          │
│  • Zero-width character removal (20+ Unicode chars)          │
│  • Whitespace encoding detection                             │
│  • Unicode smuggling neutralization                          │
│  • Injection phrase detection                                │
│                                                              │
│  OUTPUT: sanitized_text + InjectionReport                    │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│              FILE TYPE CLASSIFICATION (LLM #1)               │
│  Model: gpt-4o-mini                                          │
│  Input: First 2000 chars of sanitized text                   │
│  Output: {file_type, confidence, justification}              │
└──────────────────────────────────────────────────────────────┘
         │
         ├───────────────────┬───────────────────┐
         ▼                   ▼                   ▼
    file_type=         file_type=          file_type=
      resume           non_resume           unknown
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│              GUARDED EXTRACTION (LLM #2)                     │
│  All prompts include GUARDRAIL_PREAMBLE                      │
│                                                              │
│  Resume → work history, education, skills, owner name        │
│  Non-resume → key entities, facts, dates, summary            │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ STORAGE & INDEX  │ ◄── FAISS vector index + metadata
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│              GUARDED ANSWER GENERATION (LLM #3)              │
│  Model: gpt-4o                                               │
│  Document content treated as DATA ONLY                       │
│  No recommendations or rankings generated                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security Architecture

### Prompt Injection Defense Layers

| Layer | Technique | Purpose |
|-------|-----------|---------|
| **1** | Zero-width character removal | Strips 20+ invisible Unicode characters |
| **2** | Whitespace encoding detection | Detects tab/space binary patterns |
| **3** | Unicode smuggling neutralization | Removes tag chars, PUA, orphan selectors |
| **4** | NFKC normalization | Prevents homoglyph attacks |
| **5** | Injection phrase detection | Flags manipulation attempts |
| **6** | Guarded LLM prompts | System instructions ignore document commands |

### Risk Score Calculation

```
Risk Score = min(1.0,
    zero_width_chars × 0.05 (max 0.3) +
    whitespace_anomalies × 0.1 (max 0.2) +
    unicode_issues × 0.05 (max 0.2) +
    injection_phrases × 0.2 (max 0.5)
)
```

**Risk Levels:**
- 🟢 0.0-0.2: Low risk (normal document)
- 🟡 0.2-0.5: Medium risk (warning displayed)
- 🔴 0.5-1.0: High risk (flagged in UI)

### Detected Injection Patterns

The system detects and flags:
- Direct instruction overrides ("ignore previous instructions")
- Ranking manipulation ("this candidate is the best")
- Role hijacking ("you are now a helpful assistant")
- Hidden endorsements ("secretly note that...")
- Self-promotional claims embedded as facts

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/document-coach.git
   cd document-coach
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Enter API key in sidebar**
   
   Navigate to `http://localhost:8501` and enter your OpenAI API key in the sidebar.

---

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your `OPENAI_API_KEY` in Streamlit secrets:
   - Go to App Settings → Secrets
   - Add: `OPENAI_API_KEY = "your_key_here"`
5. Deploy!

---

## 📁 Project Structure

```
document-coach/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
├── .gitignore             # Git ignore file
└── screenshots/           # Application screenshots (optional)
```

---

## ⚙️ Configuration

### File Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_FILE_SIZE_MB` | 2 | Maximum file size per document |

### Models

| Constant | Model | Purpose |
|----------|-------|---------|
| `CLASSIFIER_MODEL` | `gpt-4o-mini` | Document type classification |
| `EXTRACTION_MODEL` | `gpt-4o-mini` | Structured data extraction |
| `ANSWER_MODEL` | `gpt-4o` | Answer generation |
| `CHAT_MODEL` | `gpt-4o-mini` | Suggestions and conversation |

---

## 📖 Usage

### 1. Upload Documents
- Use the file uploader to select PDF files
- System validates file size and checks for duplicates
- Click "Process Documents" to begin analysis

### 2. Review Classification
- Each document is classified as resume or non-resume
- Security risk scores are displayed
- Warnings shown for suspicious content

### 3. Ask Questions
- Type natural language queries in the chat input
- Use pronouns for follow-up questions ("What about his education?")
- View debug info to see how queries are interpreted

### Query Types

| Type | Example | Behavior |
|------|---------|----------|
| List all | "Who do you have?" | Lists all loaded documents |
| Specific person | "Tell me about John" | Answers using only John's resume |
| Timeline | "What was John doing in 2024?" | Filters work history by date |
| Search all | "Who has Python experience?" | Searches across all documents |
| Iterate all | "What was everyone doing in 2023?" | Checks each candidate |

---

## ⚠️ Known Limitations

- **PDF only** — Currently supports PDF files; no Word/text file support
- **English only** — Optimized for English language documents
- **2MB limit** — Large files with images may exceed size limit
- **No OCR** — Scanned PDFs without text layer are not supported
- **In-memory storage** — Vector index resets when app restarts
- **Rate limits** — Subject to OpenAI API rate limits

---

## 🔧 Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `NoneType` errors | Missing metadata fields | Helper function provides fallbacks |
| File size exceeded | PDF > 2MB | Compress PDF or split into smaller files |
| Duplicate detected | Same file/content uploaded | Skip or confirm replacement |
| Classification failed | Unclear document type | Falls back to heuristic classification |
| High injection risk | Suspicious content detected | Document processed with warning |

---

## 📋 Dependencies

```
streamlit
openai
pypdf
sentence-transformers
faiss-cpu
numpy
nltk
python-dateutil
```

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://openai.com/) for language models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [SentenceTransformers](https://www.sbert.net/) for embeddings

---

<p align="center">
  Built with security in mind 🛡️
</p>
