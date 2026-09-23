# ClauseWise — AI-Powered Legal Document Assistant

> **ClauseWise** simplifies legal documents before you sign. Get plain-English clause breakdowns, risk flagging, side-by-side contract comparison, and a lawyer consultation checklist.
> 
> *Disclaimer: ClauseWise provides informational analysis and is not a substitute for professional legal counsel.*

---

##  Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.10+, Flask, Werkzeug, Jinja2 |
| **AI / LLM** | Google Gemini API (`gemini-2.0-flash`) + Offline Regex Heuristics |
| **Document Ingestion** | `pypdf` (PDF extraction), `python-docx` (Word), `utf-8` text stream |
| **Frontend & 3D UI** | Three.js (WebGL 3D background wave & particle scene), Tailwind CSS, Vanilla JS |
| **Testing & Quality** | Pytest (51 tests), Unittest, Flake8 |
| **Security & Privacy** | In-memory ephemeral file cleanup, IP rate limiting, CSP headers |

---

## 🏛️ Design Document & Architecture

ClauseWise follows a stateless, zero-retention pipeline:

```
[ Upload (PDF/DOCX/TXT) ] 
       │
       ▼
[ Ingestion & Validation ] ──► Size limit (16MB), MIME check, ephemeral staging
       │
       ▼
[ Clause Segmentation ]   ──► Boundary extraction & semantic grouping
       │
       ▼
[ Hybrid Analysis Engine ]──► Google Gemini API (Structured JSON)
       │                      └── Fallback: Rule-based risk regex patterns
       ▼
[ Results & Output ]      ──► Plain-English summary, Risk badges (High/Med/Low),
                              Safety Score (0-100), Redlines & Lawyer Checklist
       │
       ▼
[ Ephemeral Cleanup ]     ──► Temp files immediately purged from memory & disk
```

### Core Design Decisions
1. **Hybrid Intelligence**: Uses Gemini 2.0 Flash for semantic nuance with an offline regex classifier for zero-downtime reliability.
2. **Deterministic Privacy**: Documents are processed ephemerally and deleted immediately upon request completion. No user text is stored or used for model training.
3. **Strict Non-Advisory AI**: Prompts constrain the AI to describe and categorize risks rather than offer legal opinions.
4. **Interactive 3D Background**: Three.js WebGL canvas provides depth (undulating wireframe landscape + starfield + mouse parallax) while keeping all UI cards and text clean and readable.

---

## 🚀 Quickstart

### 1. Setup & Install
```bash
git clone https://github.com/anshumansingh0408-hub/clause_wise.git
cd clause_wise
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8080
```
*(Runs with local heuristic fallback if no API key is provided).*

### 3. Run
```bash
python app.py
```
Open **http://127.0.0.1:8080** in your browser.

---

##  Testing

Run all 51 automated unit and integration tests:
```bash
pytest
```
```text
============================= 51 passed in 8.85s ==============================
```
