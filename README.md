# ClauseWise — AI-Powered Legal Document Assistant

## Overview
ClauseWise helps people understand legal documents before signing them. Upload a contract, lease, or policy and get a clause-by-clause breakdown in plain English, risk flags on unusual terms, a comparison tool for two documents, and a ready-made list of questions to bring to an actual lawyer.

**ClauseWise provides information and assistance — it does not provide legal advice and is not a substitute for a licensed attorney.**

## The Problem
Legal documents are dense and jargon-heavy. Understanding them typically requires paying a lawyer by the hour just for an initial read-through. ClauseWise closes that gap for the understanding phase, so when you do consult a lawyer, you walk in already knowing what to ask.

## Features
- **Analyze**: Clause-by-clause extraction, categorization, and risk flagging for a single document
- **Compare**: Side-by-side comparison of two documents — missing clauses, differing terms, which party each term favors
- **Ask**: Follow-up questions answered strictly from the document's own content
- **Lawyer Prep**: Auto-generated question checklist and action items

## Tech Stack
Python, Flask, Google Gemini API (gemini-2.0-flash), Tailwind CSS, pypdf, python-docx

## Responsible AI Design
Every prompt explicitly instructs Gemini to describe and flag rather than advise, recommend, or judge legal validity. Every result includes a visible disclaimer. Documents are deleted immediately after processing.

## Setup & Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
```
Open http://localhost:8080

## Testing
```bash
python -m pytest test_app.py test_analyzer.py test_comparator.py -v
```

## Code Quality Notes
- Full type hints and Google-style docstrings throughout
- Modular architecture: `app.py` (routes) / `analyzer.py` / `comparator.py` / `utils.py` (clean separation of concerns)
- `pyproject.toml` with linting config

## Security Notes
- Rate limiting per IP
- File type and size validation
- Uploaded documents deleted immediately after processing (privacy)
- Security headers (CSP, X-Frame-Options, X-XSS-Protection)
- Input sanitization on all user-provided text

## Project Structure
```
clausewise/
├── app.py                  # Flask web backend & REST API endpoints
├── analyzer.py             # Document parser, clause extractor, risk & AI analysis
├── comparator.py           # Document version comparison, redlining & metrics
├── utils.py                # Security, validation, caching & file helpers
├── test_app.py             # Integration & API route tests (unittest)
├── test_analyzer.py        # Analyzer & clause processing unit tests (unittest)
├── test_comparator.py      # Comparator & metrics unit tests (unittest)
├── requirements.txt        # Python package dependencies
├── pyproject.toml          # Test runner & code configuration
├── .env                    # Environment secrets configuration
├── Dockerfile              # Docker container definition
├── templates/
│   ├── base.html           # Master layout with theme, disclaimer banner & navigation
│   ├── index.html          # Interactive SPA upload, analysis & compare interface
│   ├── analyze.html        # Reference results view for single-document analysis
│   ├── compare.html        # Reference results view for document comparison
│   └── about.html          # Mission, methodology, taxonomy & legal disclaimer
└── uploads/                # Ephemeral in-memory storage (purged immediately)
```

## Limitations & Next Steps
- Currently English-language documents only
- Risk pattern detection is general-purpose, not jurisdiction-specific
- Architecture supports extending to e-signature platform integrations
