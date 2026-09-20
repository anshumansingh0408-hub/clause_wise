"""ClauseWise - AI-Powered Legal Document Assistant.

Intelligent contract analysis and comparison platform designed to help
individuals and organizations understand legal agreements before signing.
Parses documents, classifies individual clauses into standard commercial
categories, flags unusual or high-risk provisions, generates plain-English
summaries, and compiles actionable preparation checklists.

IMPORTANT NOTICE:
This tool assists users in understanding documents and preparing for consultations
with legal professionals. It does NOT provide legal advice and does NOT replace
a licensed attorney.
"""

import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from analyzer import (
    analyze_document,
    calculate_risk_summary,
    CLAUSE_CATEGORIES,
    LEGAL_DISCLAIMER,
    RISK_LEVELS,
)
from comparator import (
    calculate_comparison_metrics,
    compare_documents,
)
from utils import (
    allowed_file,
    call_gemini_text,
    check_rate_limit,
    clean_contract_text,
    cleanup_old_uploads,
    extract_text_from_file,
    format_confidence_badge,
    format_risk_badge,
    generate_cache_key,
    redact_preview,
    SAMPLE_CONTRACTS,
    sanitize_text_input,
    secure_filename_custom,
    validate_file_size,
)

# Load .env file if present
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

UPLOAD_FOLDER = "uploads"
MAX_FILE_SIZE_MB = 10
RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 60
CACHE_TTL = 3600
CLEANUP_INTERVAL_REQUESTS = 50
MAX_QUESTION_LEN = 300
DEFAULT_CONFIDENCE = 0.90

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_LARGE = 413
HTTP_RATE_LIMITED = 429
HTTP_SERVER_ERROR = 500

CONFIG = {
    "UPLOAD_FOLDER": UPLOAD_FOLDER,
    "MAX_FILE_SIZE_MB": MAX_FILE_SIZE_MB,
    "RATE_LIMIT_REQUESTS": RATE_LIMIT_REQUESTS,
    "RATE_LIMIT_WINDOW": RATE_LIMIT_WINDOW,
    "CACHE_TTL": CACHE_TTL,
}

SAMPLE_PREVIEWS: Dict[str, Any] = {
    "risky_lease_clause": {
        "title": "Commercial Office Lease: Unilateral Indemnification & Automatic Renewal",
        "document_type": "Commercial Lease Agreement",
        "clause_title": "Indemnity & Term Extension",
        "category": "Liability/Indemnification",
        "risk_level": "high",
        "clause_text": (
            "Tenant agrees to defend, indemnify, and hold harmless Landlord and its property managers from and against "
            "any and all claims, liabilities, losses, damages, costs, and expenses (including attorney's fees) arising out "
            "of or connected with Tenant's occupancy, regardless of whether caused in whole or in part by Landlord's active "
            "or passive negligence. Furthermore, this Lease shall automatically renew for additional successive terms of five "
            "(5) years each unless Tenant serves written notice of non-renewal via certified registered mail exactly one "
            "hundred eighty (180) days prior to the expiration of the then-current initial term."
        ),
        "plain_english_summary": (
            "This clause requires the tenant to indemnify the landlord even for damages caused by the landlord's own "
            "negligence, and locks the tenant into an automatic 5-year renewal unless notice is given exactly 180 days prior."
        ),
        "risk_analysis": {
            "score": 92,
            "flags": [
                "Uncapped unilateral indemnity including landlord negligence",
                "Strict 180-day non-renewal window with multi-year auto-renewal lock-in",
            ],
        },
        "questions_for_lawyer": [
            "How can we insert a mutual carveout excluding the landlord's gross negligence or willful misconduct?",
            "Can we negotiate the automatic renewal to require mutual written consent rather than an automatic 5-year rollover?",
        ],
    },
    "comparison_difference": {
        "title": "SaaS Master Subscription Agreement: Standard vs. Revised Redline",
        "document_type": "Software as a Service (SaaS) Agreement",
        "doc_a_name": "Standard Vendor Draft",
        "doc_b_name": "Enterprise Counterparty Redline",
        "topic": "Limitation of Liability & Termination for Convenience",
        "differing_terms": [
            {
                "term": "Aggregate Liability Cap",
                "doc_a_says": "Vendor's aggregate liability is capped at fees paid in the prior twelve (12) months.",
                "doc_b_says": "Vendor's aggregate liability is capped at 5x total contract value, with uncapped liability for data breaches.",
                "favors": "Doc B favors Customer / Doc A favors Vendor",
                "significance": "High",
            },
            {
                "term": "Termination for Convenience",
                "doc_a_says": "Neither party may terminate for convenience prior to initial 3-year term expiration.",
                "doc_b_says": "Customer may terminate at any time upon thirty (30) days prior written notice with prorated refund.",
                "favors": "Doc B favors Customer",
                "significance": "Significant",
            },
        ],
        "key_discussion_points": [
            "The counterparty eliminated the 12-month fee liability cap and introduced a supercap for data breaches.",
            "Customer added a 30-day termination for convenience with a prorated refund requirement.",
        ],
    },
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TPL_DIR = os.path.join(_BASE_DIR, "templates") if os.path.isdir(os.path.join(_BASE_DIR, "templates")) else os.path.join(_BASE_DIR, "clausewise", "templates")
_ST_DIR = os.path.join(_BASE_DIR, "static") if os.path.isdir(os.path.join(_BASE_DIR, "static")) else os.path.join(_BASE_DIR, "clausewise", "static")

app = Flask(__name__, template_folder=_TPL_DIR, static_folder=_ST_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
os.makedirs(CONFIG["UPLOAD_FOLDER"], exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# In-memory stores
JOBS: Dict[str, Any] = {}
RATE_LIMIT_TRACKER: Dict[str, List[float]] = {}
CACHE: Dict[str, Any] = {}
REQUEST_COUNTER = 0


# ============================================================================
# TEMPLATE CONTEXT & MIDDLEWARE
# ============================================================================

@app.context_processor
def inject_globals() -> Dict[str, Any]:
    """Injects common constants into Jinja template contexts.

    Returns:
        Dict[str, Any]: Dictionary of global variables available to templates.
    """
    return {
        "CLAUSE_CATEGORIES": CLAUSE_CATEGORIES,
        "RISK_LEVELS": RISK_LEVELS,
        "SAMPLE_CONTRACTS": SAMPLE_CONTRACTS,
        "LEGAL_DISCLAIMER": LEGAL_DISCLAIMER,
    }


@app.before_request
def start_timer() -> None:
    """Starts request timer for tracking response latency."""
    g.start = time.time()


@app.before_request
def periodic_cleanup() -> None:
    """Periodically purges stale files from the upload folder."""
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    if REQUEST_COUNTER % CLEANUP_INTERVAL_REQUESTS == 0:
        cleanup_old_uploads(CONFIG["UPLOAD_FOLDER"], max_age_seconds=CONFIG["CACHE_TTL"])


@app.after_request
def log_request_time(response: Response) -> Response:
    """Calculates request duration and appends X-Response-Time header.

    Args:
        response: Outgoing HTTP response.

    Returns:
        Response: Mutated HTTP response with latency header.
    """
    if hasattr(g, "start"):
        duration = time.time() - g.start
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
    return response


@app.after_request
def add_security_headers(response: Response) -> Response:
    """Appends security headers to all outgoing HTTP responses.

    Args:
        response: Outgoing HTTP response.

    Returns:
        Response: Response augmented with Content-Security-Policy and protective headers.
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://fonts.googleapis.com https://fonts.gstatic.com;"
        " script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com;"
        " style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com;"
        " font-src 'self' https://fonts.gstatic.com data:;"
        " img-src 'self' data: https:;"
        " connect-src 'self';"
    )
    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def handle_bad_request(e: Exception) -> Tuple[Response, int]:
    """Handles HTTP 400 Bad Request.

    Args:
        e: Caught HTTPException.

    Returns:
        Tuple[Response, int]: JSON response and HTTP 400 status.
    """
    msg = getattr(e, "description", str(e))
    return jsonify({"error": msg, "code": "BAD_REQUEST"}), HTTP_BAD_REQUEST


@app.errorhandler(404)
def handle_not_found(e: Exception) -> Tuple[Response, int]:
    """Handles HTTP 404 Not Found.

    Args:
        e: Caught HTTPException.

    Returns:
        Tuple[Response, int]: JSON response and HTTP 404 status.
    """
    return jsonify({"error": "Requested resource was not found.", "code": "NOT_FOUND"}), HTTP_NOT_FOUND


@app.errorhandler(413)
def handle_payload_too_large(e: Exception) -> Tuple[Response, int]:
    """Handles HTTP 413 Payload Too Large.

    Args:
        e: Caught HTTPException.

    Returns:
        Tuple[Response, int]: JSON response and HTTP 413 status.
    """
    return jsonify({"error": f"Uploaded file exceeds limit of {CONFIG['MAX_FILE_SIZE_MB']}MB.", "code": "FILE_TOO_LARGE"}), HTTP_TOO_LARGE


@app.errorhandler(429)
def handle_rate_limit_exceeded(e: Exception) -> Tuple[Response, int]:
    """Handles HTTP 429 Too Many Requests.

    Args:
        e: Caught HTTPException.

    Returns:
        Tuple[Response, int]: JSON response and HTTP 429 status.
    """
    return jsonify({"error": "Rate limit exceeded. Please wait a moment.", "code": "RATE_LIMITED"}), HTTP_RATE_LIMITED


@app.errorhandler(500)
def handle_server_error(e: Exception) -> Tuple[Response, int]:
    """Handles HTTP 500 Internal Server Error.

    Args:
        e: Caught HTTPException.

    Returns:
        Tuple[Response, int]: JSON response and HTTP 500 status.
    """
    return jsonify({"error": "An internal server error occurred.", "code": "INTERNAL_SERVER_ERROR"}), HTTP_SERVER_ERROR


# ============================================================================
# QA FALLBACK HELPERS
# ============================================================================

def _find_best_qa_clause(
    clauses: List[Dict[str, Any]], q_words: Set[str]
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Finds clause with highest keyword overlap for user question.

    Args:
        clauses: List of clauses.
        q_words: Set of meaningful query words.

    Returns:
        Tuple[Optional[Dict[str, Any]], int]: Best clause and score.
    """
    best_clause, best_score = None, 0
    for clause in clauses:
        clause_text = (clause.get("text", "") + " " + clause.get("title", "")).lower()
        matches = sum(1 for w in q_words if w in clause_text)
        if matches > best_score:
            best_score = matches
            best_clause = clause
    return best_clause, best_score


def _format_fallback_matched(question: str, clause: Dict[str, Any]) -> str:
    """Formats answer string for matched fallback clause.

    Args:
        question: User question.
        clause: Best matched clause dictionary.

    Returns:
        str: Formatted grounded response.
    """
    title = clause.get("title", "Relevant Provision")
    raw = redact_preview(clause.get("text", ""), 200)
    return (
        f"Based on the analyzed document:\n\n"
        f"Regarding '{question}', Section '{title}' provides:\n\"{raw}\"\n\n"
        f"Plain-English summary: {clause.get('plain_english', '')}\n\n"
        f"Note: This answer is derived solely from the document text and does not constitute formal legal advice."
    )


def generate_fallback_qa_answer(question: str, analysis_result: Dict[str, Any]) -> str:
    """Generates a grounded answer from analyzed clauses when the API is unavailable.

    Args:
        question: Sanitized user question.
        analysis_result: Previously analyzed document structure.

    Returns:
        str: Fact-based response extracted from stored clauses, with disclaimer.
    """
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    best_clause, best_score = _find_best_qa_clause(analysis_result.get("clauses", []), q_words)
    if best_clause and best_score > 0:
        return _format_fallback_matched(question, best_clause)
    return (
        f"Based on the provided document context, there are no specific terms addressing '{question}'. "
        f"Please review the full contract or consult with a licensed attorney for terms not explicitly covered. "
        f"This summary is for informational purposes only."
    )



# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.route("/")
def index() -> str:
    """Renders the landing page with document upload and contract analysis entry points.

    Returns:
        str: Rendered index.html template.
    """
    return render_template("index.html")


@app.route("/about")
def about() -> str:
    """Renders the methodology, clause categories, risk scoring, and legal disclaimer page.

    Returns:
        str: Rendered about.html template.
    """
    return render_template("about.html")


def _get_analyze_sources() -> Tuple[str, str, Optional[Response]]:
    """Retrieves document text and name from request form, files, or sample.

    Returns:
        Tuple[str, str, Optional[Response]]: Text, name, and redirect if error.
    """
    if request.method == "POST":
        file = request.files.get("document_file")
        if file and file.filename:
            try:
                return extract_text_from_file(file), file.filename, None
            except Exception as e:
                flash(f"Error reading file {file.filename}: {str(e)}", "danger")
                return "", "", redirect(url_for("index"))
        doc_text = request.form.get("document_text", "")
        name = request.form.get("document_title", "Uploaded Document").strip() or "Custom Document"
        return doc_text, name, None
    sample_key = request.args.get("sample", "")
    sample = SAMPLE_CONTRACTS.get(sample_key, SAMPLE_CONTRACTS["vendor_sla"])
    return sample["content"], sample["title"], None


@app.route("/analyze", methods=["GET", "POST"])
def analyze() -> Union[str, Response]:
    """Renders single document analysis UI for uploaded or sample contracts.

    Returns:
        Union[str, Response]: Rendered HTML or redirect on validation error.
    """
    doc_text, source_name, err_redirect = _get_analyze_sources()
    if err_redirect:
        return err_redirect
    doc_text = clean_contract_text(doc_text)
    if not doc_text:
        flash("Please provide contract text or upload a document (.pdf, .docx, .txt).", "warning")
        return redirect(url_for("index"))
    analysis = analyze_document(doc_text)
    return render_template("analyze.html", analysis=analysis, source_name=source_name, raw_text=doc_text)


def _get_demo_compare_pair() -> Tuple[str, str, str, str]:
    """Generates comparison texts for demo walkthrough.

    Returns:
        Tuple[str, str, str, str]: (text_a, name_a, text_b, name_b).
    """
    text_a = SAMPLE_CONTRACTS["mutual_nda"]["content"]
    text_b = text_a.replace(
        "two (2) years from the Effective Date", "ten (10) years from the Effective Date"
    ).replace(
        "Vendor provides no indemnification to Client",
        "Client shall indemnify and hold harmless Vendor against all liabilities without cap"
    ) + "\n\n10. EXCLUSIVE REMEDIES\nVendor's sole liability for any breach shall be limited to $50."
    return text_a, "Original NDA Draft", text_b, "Counterparty Revised Draft (Unfavorable Redline)"


def _extract_compare_post() -> Tuple[str, str, str, str]:
    """Extracts comparison files or texts from POST request.

    Returns:
        Tuple[str, str, str, str]: (text_a, name_a, text_b, name_b).
    """
    file_a, file_b = request.files.get("file_a"), request.files.get("file_b")
    text_a = extract_text_from_file(file_a) if file_a and file_a.filename else request.form.get("text_a", "")
    name_a = file_a.filename if file_a and file_a.filename else request.form.get("title_a", "Version 1")
    text_b = extract_text_from_file(file_b) if file_b and file_b.filename else request.form.get("text_b", "")
    name_b = file_b.filename if file_b and file_b.filename else request.form.get("title_b", "Version 2")
    return text_a, name_a, text_b, name_b


@app.route("/compare", methods=["GET", "POST"])
def compare() -> Union[str, Response]:
    """Renders document comparison UI for side-by-side contract evaluation.

    Returns:
        Union[str, Response]: Rendered compare.html page.
    """
    if request.method == "POST":
        text_a, name_a, text_b, name_b = _extract_compare_post()
    elif request.args.get("pair") == "demo":
        text_a, name_a, text_b, name_b = _get_demo_compare_pair()
    else:
        return render_template("compare.html", comparison=None)
    text_a, text_b = clean_contract_text(text_a), clean_contract_text(text_b)
    if not text_a or not text_b:
        flash("Both documents (Version A and Version B) are required for comparison.", "warning")
        return render_template("compare.html", comparison=None)
    result = compare_documents(text_a, name_a, text_b, name_b, GEMINI_API_KEY)
    return render_template("compare.html", comparison=result, name_a=name_a, name_b=name_b, text_a=text_a, text_b=text_b)


# ============================================================================
# RESTFUL API ENDPOINTS
# ============================================================================

def _enrich_badges(analysis: Dict[str, Any]) -> None:
    """Enriches analysis results with UI risk and confidence badges.

    Args:
        analysis: Analysis dictionary mutated in-place.
    """
    conf = analysis.get("confidence_score", analysis.get("metadata", {}).get("confidence_score", DEFAULT_CONFIDENCE))
    badge = format_confidence_badge(conf)
    analysis["confidence_badge"] = badge
    if "metadata" in analysis and isinstance(analysis["metadata"], dict):
        analysis["metadata"]["confidence_badge"] = badge
    for clause in analysis.get("clauses", []):
        lvl = clause.get("risk", {}).get("level", "medium") if isinstance(clause.get("risk"), dict) else clause.get("risk", "medium")
        clause["risk_badge"] = format_risk_badge(lvl)


def _safe_remove(filepath: str) -> None:
    """Safely removes a temporary file from disk if it exists.

    Args:
        filepath: Path of file to delete.
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass


def _validate_api_file(file_obj: Any) -> Optional[Tuple[Response, int]]:
    """Validates single file attachment for API analysis endpoint.

    Args:
        file_obj: Werkzeug FileStorage attachment from request.

    Returns:
        Optional[Tuple[Response, int]]: Error response tuple or None if valid.
    """
    if not file_obj or not file_obj.filename:
        return jsonify({"error": "Missing or unselected file in request.", "code": "MISSING_FILE"}), HTTP_BAD_REQUEST
    if not allowed_file(file_obj.filename):
        return jsonify({"error": "Invalid file type. Supported: .pdf, .docx, .txt.", "code": "INVALID_FILE_TYPE"}), HTTP_BAD_REQUEST
    return None


def _validate_compare_files(f_a: Any, f_b: Any) -> Optional[Tuple[Response, int]]:
    """Validates both file attachments for API comparison endpoint.

    Args:
        f_a: First file attachment from request.
        f_b: Second file attachment from request.

    Returns:
        Optional[Tuple[Response, int]]: Error response tuple or None if valid.
    """
    if not f_a or not f_b or not f_a.filename or not f_b.filename:
        return jsonify({"error": "Both 'file_a' and 'file_b' must be provided.", "code": "MISSING_FILES"}), HTTP_BAD_REQUEST
    if not allowed_file(f_a.filename) or not allowed_file(f_b.filename):
        return jsonify({"error": "Invalid or missing file attachments.", "code": "INVALID_FILE_TYPE"}), HTTP_BAD_REQUEST
    return None


def _process_analysis_upload(filepath: str, raw_filename: str, job_id: str) -> Tuple[Response, int]:
    """Runs document analysis, badges results, and stores in JOBS store.

    Args:
        filepath: Saved file path on disk.
        raw_filename: Sanitized original file name.
        job_id: Unique identifier for the analysis job.

    Returns:
        Tuple[Response, int]: JSON response and HTTP status code.
    """
    analysis = analyze_document(filepath, raw_filename, GEMINI_API_KEY)
    if not analysis or analysis.get("status") == "error":
        msg = analysis.get("error", "Analysis failed.") if analysis else "Empty analysis returned."
        JOBS[job_id] = {"status": "error", "result": None, "error": msg, "type": "analyze"}
        return jsonify({"error": msg, "code": "ANALYSIS_FAILURE"}), HTTP_SERVER_ERROR
    summary = calculate_risk_summary(analysis)
    _enrich_badges(analysis)
    JOBS[job_id] = {"status": "completed", "result": analysis, "risk_summary": summary, "error": None, "type": "analyze"}
    return jsonify({"job_id": job_id, "result": analysis, "risk_summary": summary, "disclaimer": LEGAL_DISCLAIMER}), HTTP_OK


@app.route("/api/analyze", methods=["POST"])
def api_analyze() -> Tuple[Response, int]:
    """Accepts single document upload and returns clause breakdown and checklist.

    Returns:
        Tuple[Response, int]: JSON response and HTTP status code.
    """
    if not check_rate_limit(request.remote_addr or "127.0.0.1", RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), HTTP_RATE_LIMITED
    err = _validate_api_file(request.files.get("file"))
    if err:
        return err
    file = request.files["file"]
    job_id = str(uuid.uuid4())
    raw_name = secure_filename_custom(file.filename) or f"document_{job_id[:8]}"
    filepath = os.path.join(CONFIG["UPLOAD_FOLDER"], f"{job_id}_{raw_name}")
    try:
        file.save(filepath)
        if not validate_file_size(filepath, CONFIG["MAX_FILE_SIZE_MB"]):
            return jsonify({"error": f"File exceeds {CONFIG['MAX_FILE_SIZE_MB']}MB.", "code": "FILE_TOO_LARGE"}), HTTP_BAD_REQUEST
        return _process_analysis_upload(filepath, raw_name, job_id)
    except Exception as e:
        JOBS[job_id] = {"status": "error", "result": None, "error": str(e), "type": "analyze"}
        return jsonify({"error": f"Analysis failure: {str(e)}", "code": "EXTRACTION_FAILURE"}), HTTP_SERVER_ERROR
    finally:
        _safe_remove(filepath)


def _execute_comparison_job(
    filepath_a: str, name_a: str, filepath_b: str, name_b: str, job_id: str
) -> Tuple[Response, int]:
    """Executes comparison and stores result in JOBS.

    Args:
        filepath_a: Path to first document.
        name_a: File name of first document.
        filepath_b: Path to second document.
        name_b: File name of second document.
        job_id: Unique identifier for comparison job.

    Returns:
        Tuple[Response, int]: JSON response and HTTP status code.
    """
    res = compare_documents(filepath_a, name_a, filepath_b, name_b, GEMINI_API_KEY)
    if not res or res.get("status") == "error":
        msg = res.get("error", "Comparison failed.") if res else "Empty comparison returned."
        JOBS[job_id] = {"status": "error", "result": None, "error": msg, "type": "compare"}
        return jsonify({"error": msg, "code": "COMPARISON_FAILURE"}), HTTP_SERVER_ERROR
    metrics = calculate_comparison_metrics(res)
    JOBS[job_id] = {"status": "completed", "result": res, "metrics": metrics, "error": None, "type": "compare"}
    return jsonify({"job_id": job_id, "result": res, "metrics": metrics, "disclaimer": LEGAL_DISCLAIMER}), HTTP_OK


@app.route("/api/compare", methods=["POST"])
def api_compare() -> Tuple[Response, int]:
    """Accepts two document uploads and returns differences and metrics.

    Returns:
        Tuple[Response, int]: JSON response and HTTP status code.
    """
    if not check_rate_limit(request.remote_addr or "127.0.0.1", RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), HTTP_RATE_LIMITED
    f_a, f_b = request.files.get("file_a"), request.files.get("file_b")
    err = _validate_compare_files(f_a, f_b)
    if err:
        return err
    job_id = str(uuid.uuid4())
    path_a, path_b = os.path.join(CONFIG["UPLOAD_FOLDER"], f"{job_id}_a"), os.path.join(CONFIG["UPLOAD_FOLDER"], f"{job_id}_b")
    try:
        f_a.save(path_a); f_b.save(path_b)
        if not validate_file_size(path_a, CONFIG["MAX_FILE_SIZE_MB"]) or not validate_file_size(path_b, CONFIG["MAX_FILE_SIZE_MB"]):
            return jsonify({"error": "File exceeds size limits.", "code": "FILE_TOO_LARGE"}), HTTP_BAD_REQUEST
        return _execute_comparison_job(path_a, f_a.filename, path_b, f_b.filename, job_id)
    except Exception as e:
        JOBS[job_id] = {"status": "error", "result": None, "error": str(e), "type": "compare"}
        return jsonify({"error": f"Comparison failure: {str(e)}", "code": "COMPARISON_FAILURE"}), HTTP_SERVER_ERROR
    finally:
        _safe_remove(path_a); _safe_remove(path_b)


def _build_qa_context(analysis_res: Dict[str, Any]) -> str:
    """Formats clauses into context block for Gemini grounding.

    Args:
        analysis_res: Document analysis dictionary with clauses.

    Returns:
        str: Concatenated clause context string.
    """
    clauses_context = []
    for idx, c in enumerate(analysis_res.get("clauses", []), 1):
        t, cat, txt, sm = c.get("title", f"Clause {idx}"), c.get("category", "General"), c.get("text", ""), c.get("plain_english", "")
        clauses_context.append(f"Section {idx}: {t} [{cat}]\nFull Text: {txt}\nSummary: {sm}")
    return "\n\n".join(clauses_context)


def _answer_qa_prompt(question: str, context: str, analysis: Dict[str, Any]) -> str:
    """Queries Gemini model or falls back to local heuristic for Q&A.

    Args:
        question: User question string.
        context: Extracted clause context for document.
        analysis: Full document analysis dictionary.

    Returns:
        str: Formatted QA answer.
    """
    sys_inst = (
        "You are ClauseWise Assistant. Answer the question strictly using the provided document context. "
        "State clearly if information is missing. Remind user this does not constitute formal legal advice."
    )
    prompt = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nAnswer strictly from facts above."
    ans = call_gemini_text(prompt, GEMINI_API_KEY, system_instruction=sys_inst)
    return ans if ans else generate_fallback_qa_answer(question, analysis)


@app.route("/api/ask", methods=["POST"])
def api_ask() -> Tuple[Response, int]:
    """Answers a question about an analyzed document using grounded context.

    Returns:
        Tuple[Response, int]: JSON response with answer.
    """
    if not check_rate_limit(request.remote_addr or "127.0.0.1", RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), HTTP_RATE_LIMITED
    data = request.get_json(silent=True) or {}
    job_id, raw_q = data.get("job_id", "").strip(), data.get("question", "").strip()
    if not raw_q or not job_id:
        return jsonify({"error": "Missing question or job_id parameter.", "code": "MISSING_PARAM"}), HTTP_BAD_REQUEST
    if job_id not in JOBS or JOBS[job_id].get("status") != "completed":
        return jsonify({"error": f"Completed job '{job_id}' not found.", "code": "JOB_NOT_FOUND"}), HTTP_NOT_FOUND
    sanitized_q = sanitize_text_input(raw_q, MAX_QUESTION_LEN)
    cache_key, now = generate_cache_key(job_id, sanitized_q), time.time()
    if cache_key in CACHE and now - CACHE[cache_key].get("timestamp", 0) < CONFIG["CACHE_TTL"]:
        return jsonify({"answer": CACHE[cache_key]["result"]}), HTTP_OK
    ans = _answer_qa_prompt(sanitized_q, _build_qa_context(JOBS[job_id]["result"]), JOBS[job_id]["result"])
    CACHE[cache_key] = {"result": ans, "timestamp": now}
    return jsonify({"answer": ans}), HTTP_OK


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str) -> Tuple[Response, int]:
    """Retrieves status and result for a submitted analysis or comparison job.

    Args:
        job_id: Unique UUID string identifier of the job.

    Returns:
        Tuple[Response, int]: Status dictionary and HTTP status code.
    """
    if not job_id or job_id not in JOBS:
        return jsonify({"error": f"Job ID '{job_id}' not found.", "code": "NOT_FOUND"}), HTTP_NOT_FOUND
    return jsonify(JOBS[job_id]), HTTP_OK


@app.route("/api/sample", methods=["GET"])
def api_sample() -> Tuple[Response, int]:
    """Returns realistic fictional example scenarios for homepage preview.

    Returns:
        Tuple[Response, int]: JSON response with sample scenarios.
    """
    return jsonify(SAMPLE_PREVIEWS), HTTP_OK


@app.route("/api/sample/<sample_key>", methods=["GET"])
def get_sample_contract(sample_key: str) -> Tuple[Response, int]:
    """API endpoint returning sample contract text by key.

    Args:
        sample_key: Identifier of the sample contract.

    Returns:
        Tuple[Response, int]: Sample contract JSON or 404 error.
    """
    if sample_key in SAMPLE_CONTRACTS:
        return jsonify(SAMPLE_CONTRACTS[sample_key]), HTTP_OK
    return jsonify({"error": f"Sample contract '{sample_key}' not found.", "code": "NOT_FOUND"}), HTTP_NOT_FOUND


def _build_checklist_markdown(items: List[Dict[str, Any]], title: str) -> str:
    """Builds Markdown content string for lawyer preparation checklist.

    Args:
        items: List of checklist items.
        title: Document title string.

    Returns:
        str: Formatted Markdown string.
    """
    lines = [
        f"# ClauseWise Lawyer-Prep Checklist: {title}", "",
        "> **Notice**: Generated for informational and preparatory purposes only. Does NOT constitute legal advice.", "",
        "---", ""
    ]
    for idx, item in enumerate(items, start=1):
        emoji = "🔴" if item.get("priority") == "Urgent" else "🟡"
        lines.append(f"### {idx}. [{item.get('priority', 'Review')}] {emoji} {item.get('clause_number', '§')} - {item.get('clause_title', 'Clause')}")
        lines.append(f"- **Category**: {item.get('category', 'Other')}")
        lines.append(f"- **Identified Risk/Concern**: {item.get('concern', '')}")
        lines.append(f"- **Suggested Negotiation Action**: {item.get('action_item', '')}\n")
    return "\n".join(lines)


@app.route("/api/export-checklist", methods=["POST"])
def export_checklist() -> Response:
    """Exports generated lawyer-prep checklist as a downloadable Markdown document.

    Returns:
        Response: Flask send_file response streaming markdown buffer.
    """
    data = request.get_json(silent=True) or {}
    content = _build_checklist_markdown(data.get("items", []), data.get("title", "Contract Analysis"))
    return send_file(
        io.BytesIO(content.encode("utf-8")),
        as_attachment=True,
        download_name="ClauseWise_Lawyer_Prep_Checklist.md",
        mimetype="text/markdown",
    )


@app.route("/health")
def health() -> Tuple[Response, int]:
    """Health check endpoint for container and uptime monitoring.

    Returns:
        Tuple[Response, int]: JSON health status and HTTP 200.
    """
    return jsonify({"status": "healthy", "service": "clausewise"}), HTTP_OK


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
