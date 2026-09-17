"""ClauseWise - AI-Powered Legal Document Assistant.

ClauseWise is an intelligent contract analysis and comparison platform designed to help
individuals and organizations understand legal agreements before signing. It parses
documents, classifies individual clauses into standard commercial categories, flags
unusual or high-risk provisions, generates plain-English summaries, and compiles
actionable preparation checklists for consultations with licensed attorneys.

Architecture Overview:
- analyzer.py: Document parsing, clause extraction, rule-based heuristics, and Gemini AI analysis.
- comparator.py: Side-by-side contract comparison, redline difference tracking, and negotiation points.
- utils.py: File validation, sanitization, rate limiting, badge formatting, caching, and cleanup helpers.
- app.py (Flask Backend): RESTful API endpoints, rate limiting, in-memory job/cache management,
  and server-rendered UI integration.

IMPORTANT NOTICE:
This tool assists users in understanding documents and preparing for consultations with
legal professionals. It does NOT provide legal advice and does NOT replace a licensed attorney.
"""

import os
import time
import uuid
import io
import re
import requests
from flask import Flask, render_template, request, jsonify, send_file, g, redirect, url_for, flash

from analyzer import (
    analyze_document, calculate_risk_summary, LEGAL_DISCLAIMER,
    call_gemini_api, CLAUSE_CATEGORIES, RISK_LEVELS
)
from comparator import compare_documents, calculate_comparison_metrics
from utils import (
    allowed_file, secure_filename_custom, generate_cache_key,
    validate_file_size, format_confidence_badge, 
    format_risk_badge, check_rate_limit, sanitize_text_input,
    cleanup_old_uploads, redact_preview,
    extract_text_from_file, clean_contract_text, SAMPLE_CONTRACTS
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

# === CONFIG ===
CONFIG = {
    "UPLOAD_FOLDER": "uploads",
    "MAX_FILE_SIZE_MB": 10,
    "RATE_LIMIT_REQUESTS": 15,
    "RATE_LIMIT_WINDOW": 60,
    "CACHE_TTL": 3600
}

os.makedirs(CONFIG["UPLOAD_FOLDER"], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# === IN-MEMORY STORES ===
JOBS = {}  # job_id -> {"status": str, "result": dict, "error": str, "type": "analyze"|"compare"}
RATE_LIMIT_TRACKER = {}  # ip -> list of timestamps
CACHE = {}  # cache_key -> {"result": dict, "timestamp": float}
REQUEST_COUNTER = 0


# ============================================================================
# TEMPLATE CONTEXT PROCESSOR
# ============================================================================

@app.context_processor
def inject_globals():
    """Injects common constants into Jinja template contexts."""
    return {
        "CLAUSE_CATEGORIES": CLAUSE_CATEGORIES,
        "RISK_LEVELS": RISK_LEVELS,
        "SAMPLE_CONTRACTS": SAMPLE_CONTRACTS,
        "LEGAL_DISCLAIMER": LEGAL_DISCLAIMER
    }


# ============================================================================
# EFFICIENCY & SECURITY MIDDLEWARE
# ============================================================================

@app.before_request
def start_timer():
    """Starts request timer for tracking response latency."""
    g.start = time.time()


@app.before_request
def periodic_cleanup():
    """Periodically purges stale files from the upload folder."""
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    if REQUEST_COUNTER % 50 == 0:
        cleanup_old_uploads(CONFIG["UPLOAD_FOLDER"], max_age_seconds=CONFIG["CACHE_TTL"])


@app.after_request
def log_request_time(response):
    """Calculates request duration and appends X-Response-Time header."""
    if hasattr(g, "start"):
        duration = time.time() - g.start
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
    return response


@app.after_request
def add_security_headers(response):
    """Appends security headers to all outgoing HTTP responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


# ============================================================================
# ERROR HANDLERS (Consistent JSON error responses)
# ============================================================================

@app.errorhandler(400)
def handle_bad_request(e):
    """Handles HTTP 400 Bad Request."""
    msg = getattr(e, "description", str(e))
    return jsonify({"error": msg, "code": "BAD_REQUEST"}), 400


@app.errorhandler(404)
def handle_not_found(e):
    """Handles HTTP 404 Not Found."""
    return jsonify({"error": "Requested resource was not found.", "code": "NOT_FOUND"}), 404


@app.errorhandler(413)
def handle_payload_too_large(e):
    """Handles HTTP 413 Payload Too Large."""
    return jsonify({
        "error": f"Uploaded file exceeds the maximum limit of {CONFIG['MAX_FILE_SIZE_MB']}MB.",
        "code": "FILE_TOO_LARGE"
    }), 413


@app.errorhandler(429)
def handle_rate_limit_exceeded(e):
    """Handles HTTP 429 Too Many Requests."""
    return jsonify({"error": "Rate limit exceeded. Please wait a moment.", "code": "RATE_LIMITED"}), 429


@app.errorhandler(500)
def handle_server_error(e):
    """Handles HTTP 500 Internal Server Error."""
    return jsonify({"error": "An internal server error occurred.", "code": "INTERNAL_SERVER_ERROR"}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_gemini_text(prompt: str, api_key: str, system_instruction: str = None) -> str:
    """Calls Gemini API directly and returns an unstructured text response.

    Args:
        prompt: User prompt containing document context and follow-up question.
        api_key: Google Gemini API key.
        system_instruction: Optional system instruction for grounding the model.

    Returns:
        str: Response text from Gemini, or None if unavailable.
    """
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception:
        pass
    return None


def generate_fallback_qa_answer(question: str, analysis_result: dict) -> str:
    """Generates a grounded answer from analyzed clauses when the API is unavailable.

    Args:
        question: Sanitized user question.
        analysis_result: Previously analyzed document structure.

    Returns:
        str: Fact-based response extracted from stored clauses, with disclaimer.
    """
    question_lower = question.lower()
    q_words = set(re.findall(r"\b\w{4,}\b", question_lower))

    clauses = analysis_result.get("clauses", [])
    best_clause = None
    best_score = 0

    for clause in clauses:
        clause_text = (clause.get("text", "") + " " + clause.get("title", "")).lower()
        matches = sum(1 for w in q_words if w in clause_text)
        if matches > best_score:
            best_score = matches
            best_clause = clause

    if best_clause and best_score > 0:
        c_title = best_clause.get("title", "Relevant Provision")
        c_summary = best_clause.get("plain_english", "")
        c_raw = redact_preview(best_clause.get("text", ""), 200)
        answer = (
            f"Based on the analyzed document:\n\n"
            f"Regarding '{question}', Section '{c_title}' provides:\n\"{c_raw}\"\n\n"
            f"Plain-English summary: {c_summary}\n\n"
            f"Note: This answer is derived solely from the document text and does not constitute formal legal advice."
        )
    else:
        answer = (
            f"Based on the provided document context, there are no specific terms addressing '{question}'. "
            f"Please review the full contract or consult with a licensed attorney for terms not explicitly covered. "
            f"This summary is for informational purposes only."
        )
    return answer


# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.route("/")
def index():
    """Renders the landing page with document upload and contract analysis entry points."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Renders the methodology, clause categories, risk scoring, and legal disclaimer page."""
    return render_template("about.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """Renders single document analysis UI for uploaded or sample contracts."""
    document_text = ""
    source_name = "Pasted Contract Text"

    if request.method == "POST":
        file = request.files.get("document_file")
        if file and file.filename:
            try:
                document_text = extract_text_from_file(file)
                source_name = file.filename
            except Exception as e:
                flash(f"Error reading file {file.filename}: {str(e)}", "danger")
                return redirect(url_for("index"))
        else:
            document_text = request.form.get("document_text", "")
            source_name = request.form.get("document_title", "Uploaded Document").strip() or "Custom Document"

    elif request.method == "GET":
        sample_key = request.args.get("sample", "")
        if sample_key in SAMPLE_CONTRACTS:
            sample = SAMPLE_CONTRACTS[sample_key]
            document_text = sample["content"]
            source_name = sample["title"]
        else:
            sample = SAMPLE_CONTRACTS["vendor_sla"]
            document_text = sample["content"]
            source_name = sample["title"]

    document_text = clean_contract_text(document_text)

    if not document_text:
        flash("Please provide contract text or upload a document (.pdf, .docx, .txt).", "warning")
        return redirect(url_for("index"))

    analysis_result = analyze_document(document_text)

    return render_template(
        "analyze.html",
        analysis=analysis_result,
        source_name=source_name,
        raw_text=document_text
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Renders document comparison UI for side-by-side contract evaluation."""
    text_a = ""
    text_b = ""
    name_a = "Version A"
    name_b = "Version B"

    if request.method == "POST":
        file_a = request.files.get("file_a")
        if file_a and file_a.filename:
            try:
                text_a = extract_text_from_file(file_a)
                name_a = file_a.filename
            except Exception as e:
                flash(f"Error reading Document A: {str(e)}", "danger")
                return redirect(url_for("compare"))
        else:
            text_a = request.form.get("text_a", "")
            name_a = request.form.get("title_a", "Version 1")

        file_b = request.files.get("file_b")
        if file_b and file_b.filename:
            try:
                text_b = extract_text_from_file(file_b)
                name_b = file_b.filename
            except Exception as e:
                flash(f"Error reading Document B: {str(e)}", "danger")
                return redirect(url_for("compare"))
        else:
            text_b = request.form.get("text_b", "")
            name_b = request.form.get("title_b", "Version 2")

    elif request.method == "GET":
        sample_pair = request.args.get("pair")
        if sample_pair == "demo":
            text_a = SAMPLE_CONTRACTS["mutual_nda"]["content"]
            name_a = "Original NDA Draft"
            text_b = text_a.replace(
                "two (2) years from the Effective Date",
                "ten (10) years from the Effective Date"
            ).replace(
                "Vendor provides no indemnification to Client",
                "Client shall indemnify and hold harmless Vendor against all liabilities without cap"
            ) + "\n\n10. EXCLUSIVE REMEDIES\nVendor's sole liability for any breach shall be limited to $50."
            name_b = "Counterparty Revised Draft (Unfavorable Redline)"
        else:
            return render_template("compare.html", comparison=None)

    text_a = clean_contract_text(text_a)
    text_b = clean_contract_text(text_b)

    if not text_a or not text_b:
        flash("Both documents (Version A and Version B) are required for comparison.", "warning")
        return render_template("compare.html", comparison=None)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    comparison_result = compare_documents(text_a, name_a, text_b, name_b, api_key)

    return render_template(
        "compare.html",
        comparison=comparison_result,
        name_a=name_a,
        name_b=name_b,
        text_a=text_a,
        text_b=text_b
    )


# ============================================================================
# RESTFUL API ENDPOINTS
# ============================================================================

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Accepts a single document upload, runs it through the analysis pipeline,
    returns clause breakdown, risk summary, lawyer questions, and action checklist.

    Returns:
        JSON: job_id, result, risk_summary, and legal disclaimer.
    """
    client_ip = request.remote_addr or "127.0.0.1"
    if not check_rate_limit(client_ip, RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), 429

    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' parameter in request.", "code": "MISSING_FILE"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected for upload.", "code": "NO_FILE_SELECTED"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type. Only PDF (.pdf), Word (.docx), and Text (.txt) files are supported.",
            "code": "INVALID_FILE_TYPE"
        }), 400

    job_id = str(uuid.uuid4())
    raw_filename = secure_filename_custom(file.filename) or f"document_{job_id[:8]}"
    saved_name = f"{job_id}_{raw_filename}"
    filepath = os.path.join(CONFIG["UPLOAD_FOLDER"], saved_name)

    try:
        file.save(filepath)

        if not validate_file_size(filepath, CONFIG["MAX_FILE_SIZE_MB"]):
            return jsonify({
                "error": f"File exceeds maximum allowed size of {CONFIG['MAX_FILE_SIZE_MB']}MB.",
                "code": "FILE_TOO_LARGE"
            }), 400

        analysis_result = analyze_document(filepath, raw_filename, GEMINI_API_KEY)

        if not analysis_result or analysis_result.get("status") == "error":
            err_msg = analysis_result.get("error", "Analysis failed.") if analysis_result else "Empty analysis returned."
            JOBS[job_id] = {
                "status": "error",
                "result": None,
                "error": err_msg,
                "type": "analyze"
            }
            return jsonify({"error": err_msg, "code": "ANALYSIS_FAILURE"}), 500

        # Calculate risk summary
        risk_summary = calculate_risk_summary(analysis_result)

        # Append confidence badge
        confidence_val = analysis_result.get("confidence_score")
        if confidence_val is None:
            confidence_val = analysis_result.get("metadata", {}).get("confidence_score", 0.90)
        confidence_badge = format_confidence_badge(confidence_val)
        analysis_result["confidence_badge"] = confidence_badge
        if "metadata" in analysis_result and isinstance(analysis_result["metadata"], dict):
            analysis_result["metadata"]["confidence_badge"] = confidence_badge

        # Append risk badges per clause
        for clause in analysis_result.get("clauses", []):
            if isinstance(clause, dict):
                r_info = clause.get("risk", {})
                if isinstance(r_info, dict):
                    lvl = r_info.get("level", "medium")
                    badge = format_risk_badge(lvl)
                    r_info["badge"] = badge
                    clause["risk_badge"] = badge
                elif isinstance(r_info, str):
                    badge = format_risk_badge(r_info)
                    clause["risk_badge"] = badge

        # Store in JOBS store
        JOBS[job_id] = {
            "status": "completed",
            "result": analysis_result,
            "risk_summary": risk_summary,
            "error": None,
            "type": "analyze"
        }

        return jsonify({
            "job_id": job_id,
            "result": analysis_result,
            "risk_summary": risk_summary,
            "disclaimer": LEGAL_DISCLAIMER
        }), 200

    except Exception as e:
        JOBS[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e),
            "type": "analyze"
        }
        return jsonify({"error": f"Document extraction or analysis failure: {str(e)}", "code": "EXTRACTION_FAILURE"}), 500

    finally:
        # Privacy: immediately purge the uploaded file after processing
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Accepts two document uploads, runs comparison pipeline,
    returns differences, missing clauses, and discussion points.

    Returns:
        JSON: job_id, result, metrics, and legal disclaimer.
    """
    client_ip = request.remote_addr or "127.0.0.1"
    if not check_rate_limit(client_ip, RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), 429

    if "file_a" not in request.files or "file_b" not in request.files:
        return jsonify({"error": "Both 'file_a' and 'file_b' must be provided in request.", "code": "MISSING_FILES"}), 400

    file_a = request.files["file_a"]
    file_b = request.files["file_b"]

    if not file_a or not file_a.filename or not file_b or not file_b.filename:
        return jsonify({"error": "Both file_a and file_b must be selected.", "code": "NO_FILE_SELECTED"}), 400

    if not allowed_file(file_a.filename) or not allowed_file(file_b.filename):
        return jsonify({
            "error": "Invalid file type. Supported formats are PDF (.pdf), Word (.docx), and Text (.txt).",
            "code": "INVALID_FILE_TYPE"
        }), 400

    job_id = str(uuid.uuid4())
    name_a = secure_filename_custom(file_a.filename) or f"doc_a_{job_id[:8]}"
    name_b = secure_filename_custom(file_b.filename) or f"doc_b_{job_id[:8]}"
    filepath_a = os.path.join(CONFIG["UPLOAD_FOLDER"], f"{job_id}_a_{name_a}")
    filepath_b = os.path.join(CONFIG["UPLOAD_FOLDER"], f"{job_id}_b_{name_b}")

    try:
        file_a.save(filepath_a)
        file_b.save(filepath_b)

        if not validate_file_size(filepath_a, CONFIG["MAX_FILE_SIZE_MB"]) or not validate_file_size(filepath_b, CONFIG["MAX_FILE_SIZE_MB"]):
            return jsonify({
                "error": f"One or both files exceed the maximum allowed size of {CONFIG['MAX_FILE_SIZE_MB']}MB.",
                "code": "FILE_TOO_LARGE"
            }), 400

        comparison_result = compare_documents(filepath_a, name_a, filepath_b, name_b, GEMINI_API_KEY)

        if not comparison_result or comparison_result.get("status") == "error":
            err_msg = comparison_result.get("error", "Document comparison failed.") if comparison_result else "Empty comparison returned."
            JOBS[job_id] = {
                "status": "error",
                "result": None,
                "error": err_msg,
                "type": "compare"
            }
            return jsonify({"error": err_msg, "code": "COMPARISON_FAILURE"}), 500

        metrics = calculate_comparison_metrics(comparison_result)

        JOBS[job_id] = {
            "status": "completed",
            "result": comparison_result,
            "metrics": metrics,
            "error": None,
            "type": "compare"
        }

        return jsonify({
            "job_id": job_id,
            "result": comparison_result,
            "metrics": metrics,
            "disclaimer": LEGAL_DISCLAIMER
        }), 200

    except Exception as e:
        JOBS[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e),
            "type": "compare"
        }
        return jsonify({"error": f"Comparison execution failure: {str(e)}", "code": "COMPARISON_FAILURE"}), 500

    finally:
        for fp in (filepath_a, filepath_b):
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Answers a follow-up question about a previously analyzed document,
    using the stored analysis result as context.

    Returns:
        JSON: {"answer": str}
    """
    client_ip = request.remote_addr or "127.0.0.1"
    if not check_rate_limit(client_ip, RATE_LIMIT_TRACKER, CONFIG["RATE_LIMIT_REQUESTS"], CONFIG["RATE_LIMIT_WINDOW"]):
        return jsonify({"error": "Rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), 429

    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id", "").strip()
    raw_question = data.get("question", "").strip()

    if not raw_question:
        return jsonify({"error": "Question cannot be empty.", "code": "EMPTY_QUESTION"}), 400

    if not job_id:
        return jsonify({"error": "Missing 'job_id' parameter.", "code": "MISSING_JOB_ID"}), 400

    if job_id not in JOBS:
        return jsonify({"error": f"Job '{job_id}' not found.", "code": "JOB_NOT_FOUND"}), 404

    job = JOBS[job_id]
    if job.get("status") != "completed" or not job.get("result"):
        return jsonify({"error": "Job analysis is not completed or has no valid results.", "code": "JOB_INCOMPLETE"}), 400

    if len(raw_question) > 300:
        return jsonify({"error": "Question exceeds maximum length of 300 characters.", "code": "QUESTION_TOO_LONG"}), 400

    sanitized_question = sanitize_text_input(raw_question, 300)
    if not sanitized_question:
        return jsonify({"error": "Question contains only invalid or control characters.", "code": "EMPTY_QUESTION"}), 400

    # Cache lookup
    cache_key = generate_cache_key(job_id, sanitized_question)
    now = time.time()
    if cache_key in CACHE:
        cache_entry = CACHE[cache_key]
        if now - cache_entry.get("timestamp", 0) < CONFIG["CACHE_TTL"]:
            return jsonify({"answer": cache_entry["result"]}), 200

    # Assemble grounding context from stored analysis
    analysis_res = job["result"]
    clauses_context = []
    for idx, c in enumerate(analysis_res.get("clauses", []), 1):
        title = c.get("title", f"Clause {idx}")
        cat = c.get("category", "General")
        text = c.get("text", "")
        summary = c.get("plain_english", "")
        clauses_context.append(f"Section {idx}: {title} [{cat}]\nFull Text: {text}\nSummary: {summary}")

    context_str = "\n\n".join(clauses_context)

    system_instruction = (
        "You are ClauseWise Assistant, a document analysis assistant. Answer the user's question "
        "strictly and ONLY based on the provided document context below. Do not speculate or assume terms "
        "not stated in the document. If the answer cannot be determined from the document, state clearly "
        "that the document does not contain this information. Remind the user that your answer is for "
        "informational understanding only and does not constitute formal legal advice."
    )
    prompt = f"""DOCUMENT CONTEXT:
{context_str}

USER QUESTION:
{sanitized_question}

INSTRUCTIONS:
1. Answer strictly using ONLY facts from the document context above.
2. Decline to speculate on unstated terms or outside legal rules.
3. Conclude with a reminder that this explanation reflects document contents only and does not constitute formal legal advice.
"""

    answer = call_gemini_text(prompt, GEMINI_API_KEY, system_instruction=system_instruction)

    if not answer:
        answer = generate_fallback_qa_answer(sanitized_question, analysis_res)

    # Store in cache
    CACHE[cache_key] = {"result": answer, "timestamp": now}

    return jsonify({"answer": answer}), 200


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    """Retrieves the status and result for a submitted analysis or comparison job.

    Args:
        job_id: Unique UUID string identifier of the job.

    Returns:
        JSON: Status object with status, result, error, and job type.
    """
    if not job_id or job_id not in JOBS:
        return jsonify({"error": f"Job ID '{job_id}' not found.", "code": "NOT_FOUND"}), 404

    return jsonify(JOBS[job_id]), 200


@app.route("/api/sample", methods=["GET"])
def api_sample():
    """Returns two realistic fictional example scenarios for homepage preview.

    Returns:
        JSON: Scenarios demonstrating a risky lease clause and a comparison difference.
    """
    samples = {
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
                    "Strict 180-day non-renewal window with multi-year auto-renewal lock-in"
                ]
            },
            "questions_for_lawyer": [
                "How can we insert a mutual carveout excluding the landlord's gross negligence or willful misconduct?",
                "Can we negotiate the automatic renewal to require mutual written consent rather than an automatic 5-year rollover?"
            ]
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
                    "significance": "High"
                },
                {
                    "term": "Termination for Convenience",
                    "doc_a_says": "Neither party may terminate for convenience prior to initial 3-year term expiration.",
                    "doc_b_says": "Customer may terminate at any time upon thirty (30) days prior written notice with prorated refund.",
                    "favors": "Doc B favors Customer",
                    "significance": "Significant"
                }
            ],
            "key_discussion_points": [
                "The counterparty eliminated the 12-month fee liability cap and introduced a supercap for data breaches.",
                "Customer added a 30-day termination for convenience with a prorated refund requirement."
            ]
        }
    }
    return jsonify(samples), 200


@app.route("/api/sample/<sample_key>", methods=["GET"])
def get_sample_contract(sample_key: str):
    """API endpoint returning sample contract text by key.

    Args:
        sample_key: Identifier of the sample contract.

    Returns:
        JSON: Sample contract dictionary or 404 error.
    """
    if sample_key in SAMPLE_CONTRACTS:
        return jsonify(SAMPLE_CONTRACTS[sample_key]), 200
    return jsonify({"error": f"Sample contract '{sample_key}' not found.", "code": "NOT_FOUND"}), 404


@app.route("/api/export-checklist", methods=["POST"])
def export_checklist():
    """Exports the generated lawyer-prep checklist as a downloadable Markdown document."""
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    title = data.get("title", "Contract Analysis")

    lines = [
        f"# ClauseWise Lawyer-Prep Checklist: {title}",
        "",
        "> **Notice**: This document is generated for informational and preparatory purposes only.",
        "> It does NOT constitute legal advice. Review these points with your qualified attorney.",
        "",
        "---",
        ""
    ]

    for idx, item in enumerate(items, start=1):
        priority_emoji = "🔴" if item.get("priority") == "Urgent" else "🟡"
        lines.append(f"### {idx}. [{item.get('priority', 'Review')}] {priority_emoji} {item.get('clause_number', '§')} - {item.get('clause_title', 'Clause')}")
        lines.append(f"- **Category**: {item.get('category', 'Other')}")
        lines.append(f"- **Identified Risk/Concern**: {item.get('concern', '')}")
        lines.append(f"- **Suggested Negotiation Action**: {item.get('action_item', '')}")
        lines.append("")

    content = "\n".join(lines)
    buffer = io.BytesIO(content.encode("utf-8"))
    return send_file(
        buffer,
        as_attachment=True,
        download_name="ClauseWise_Lawyer_Prep_Checklist.md",
        mimetype="text/markdown"
    )


@app.route("/health")
def health():
    """Health check endpoint for Docker container and monitoring uptime."""
    return jsonify({"status": "healthy", "service": "clausewise"}), 200


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
