"""Utility functions for file handling, caching, validation, and
security helpers used across the ClauseWise application.
"""

import hashlib
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional
import requests

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10
BYTES_PER_MB = 1024 * 1024
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * BYTES_PER_MB

DEFAULT_MAX_AGE_SECONDS = 3600
DEFAULT_PREVIEW_MAX_CHARS = 300
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.60
PERCENTAGE_MULTIPLIER = 100

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEXT_MAX_TOKENS = 2048
HTTP_STATUS_OK = 200

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

CONFIG = {
    "ALLOWED_EXTENSIONS": ALLOWED_EXTENSIONS,
    "MAX_FILE_SIZE_MB": MAX_FILE_SIZE_MB,
    "TIMEOUT_SECONDS": DEFAULT_TIMEOUT_SECONDS,
    "TEMPERATURE": DEFAULT_TEMPERATURE,
    "MAX_OUTPUT_TOKENS": DEFAULT_MAX_OUTPUT_TOKENS,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Checks if filename has an extension in ALLOWED_EXTENSIONS (case-insensitive).

    Args:
        filename: Name or path of the file to inspect.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.

    Raises:
        None.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def _clean_traversal_tokens(filename: str) -> str:
    """Helper to remove directory traversal tokens from a filename string.

    Args:
        filename: Filename string to clean.

    Returns:
        str: Cleaned string without traversal tokens.

    Raises:
        None.
    """
    sanitized = filename
    while "../" in sanitized or "..\\" in sanitized:
        sanitized = sanitized.replace("../", "").replace("..\\", "")
    sanitized = sanitized.replace("/", "").replace("\\", "")
    return sanitized


def secure_filename_custom(filename: str) -> str:
    """Sanitizes a filename to prevent directory traversal and unsafe characters.

    Args:
        filename: The original filename or path to sanitize.

    Returns:
        str: A sanitized, safe filename containing only allowed characters.

    Raises:
        None.
    """
    if not filename:
        return ""
    sanitized = _clean_traversal_tokens(filename)
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "", sanitized)
    while ".." in sanitized:
        sanitized = sanitized.replace("..", ".")
    return sanitized.strip(". ")


def generate_cache_key(*args: str) -> str:
    """Generates an MD5 hash cache key from any number of string arguments.

    Args:
        *args: Variable number of string arguments.

    Returns:
        str: Hexadecimal MD5 hash representing the cache key.

    Raises:
        None.
    """
    hasher = hashlib.md5()
    for arg in args:
        hasher.update(str(arg).encode("utf-8"))
    return hasher.hexdigest()


def validate_file_size(filepath: str, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """Checks if a file at filepath is within the max size limit in megabytes.

    Args:
        filepath: Path to the file to check.
        max_mb: Maximum allowed file size in megabytes. Defaults to MAX_FILE_SIZE_MB.

    Returns:
        bool: True if the file exists and its size is <= max_mb, False otherwise.

    Raises:
        None.
    """
    if not filepath or not os.path.isfile(filepath):
        return False
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes <= max_mb * BYTES_PER_MB
    except OSError:
        return False


def format_confidence_badge(score: float) -> Dict[str, Any]:
    """Converts a confidence score (0.0-1.0) into display information.

    Args:
        score: Numerical confidence score between 0.0 and 1.0.

    Returns:
        Dict[str, Any]: Dictionary containing label, color, and percentage.

    Raises:
        None.
    """
    pct = int(round(score * PERCENTAGE_MULTIPLIER))
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return {"label": "High Confidence", "color": "green", "percentage": pct}
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return {"label": "Medium Confidence", "color": "yellow", "percentage": pct}
    return {
        "label": "Low Confidence - Review Carefully",
        "color": "red",
        "percentage": pct,
    }


def format_risk_badge(risk_level: str) -> Dict[str, str]:
    """Converts a risk_level string into display information.

    Args:
        risk_level: Risk level string ("low", "medium", or "high").

    Returns:
        Dict[str, str]: Dictionary containing label, color, and icon.

    Raises:
        None.
    """
    level = (risk_level or "").strip().lower()
    badges = {
        "low": {"label": "Standard", "color": "green", "icon": "✓"},
        "medium": {"label": "Worth Reviewing", "color": "yellow", "icon": "⚠"},
        "high": {"label": "Needs Attention", "color": "red", "icon": "⚠"},
    }
    return badges.get(level, {"label": "Worth Reviewing", "color": "yellow", "icon": "⚠"})


def check_rate_limit(
    ip: str, tracker: Dict[str, List[float]], max_requests: int, window_seconds: int
) -> bool:
    """Generic rate limiter that checks and updates timestamps in place.

    Args:
        ip: Client IP address or unique requester identifier.
        tracker: Dictionary mapping IP strings to lists of float timestamps.
        max_requests: Maximum number of allowed requests in the time window.
        window_seconds: Window duration in seconds.

    Returns:
        bool: True if request is allowed, False if rate limited.

    Raises:
        None.
    """
    now = time.time()
    cutoff = now - window_seconds
    timestamps = [t for t in tracker.get(ip, []) if t > cutoff]
    tracker[ip] = timestamps
    if len(timestamps) < max_requests:
        tracker[ip].append(now)
        return True
    return False


def sanitize_text_input(text: str, max_length: int) -> str:
    """Strips and truncates text input, removing null bytes and control characters.

    Args:
        text: Input string to sanitize.
        max_length: Maximum permitted length in characters.

    Returns:
        str: Sanitized and truncated string.

    Raises:
        None.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text).strip()
    if max_length is not None and max_length >= 0:
        cleaned = cleaned[:max_length]
    return cleaned


def _clean_single_file(entry_path: str, now: float, max_age: int) -> bool:
    """Helper to remove a file if it exceeds maximum age.

    Args:
        entry_path: Path to the target file.
        now: Current epoch timestamp.
        max_age: Maximum file age in seconds.

    Returns:
        bool: True if removed successfully, False otherwise.

    Raises:
        None.
    """
    try:
        if os.path.isfile(entry_path) and (now - os.path.getmtime(entry_path)) > max_age:
            os.remove(entry_path)
            return True
    except OSError:
        pass
    return False


def cleanup_old_uploads(
    upload_folder: str, max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS
) -> int:
    """Deletes files in upload_folder older than max_age_seconds for privacy.

    Args:
        upload_folder: Path to directory containing uploaded files.
        max_age_seconds: Maximum allowed age in seconds. Defaults to 3600.

    Returns:
        int: Number of files successfully deleted.

    Raises:
        None.
    """
    if not upload_folder or not os.path.isdir(upload_folder):
        return 0
    now, count = time.time(), 0
    try:
        for entry in os.listdir(upload_folder):
            if _clean_single_file(os.path.join(upload_folder, entry), now, max_age_seconds):
                count += 1
    except OSError:
        pass
    return count


def redact_preview(text: str, max_chars: int = DEFAULT_PREVIEW_MAX_CHARS) -> str:
    """Truncates document text to max_chars for safe preview display.

    Args:
        text: Document text to preview.
        max_chars: Maximum character count before truncation.

    Returns:
        str: Safe preview text, with ellipsis added if truncated.

    Raises:
        None.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def clean_contract_text(text: str) -> str:
    """Normalizes whitespace and standardizes line breaks in contract text.

    Args:
        text: Raw contract text to normalize.

    Returns:
        str: Normalized text with consistent newlines and stripped whitespace.

    Raises:
        None.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# FILE EXTRACTION HELPERS
# ============================================================================

def _extract_pdf_pages(stream: io.BytesIO) -> str:
    """Extracts text from PDF stream using PdfReader.

    Args:
        stream: Byte stream of the PDF file.

    Returns:
        str: Joined text extracted from all PDF pages.

    Raises:
        RuntimeError: If pypdf is not installed.
    """
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Unable to process PDF files.")
    reader = PdfReader(stream)
    extracted = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(extracted)


def _extract_docx_paragraphs(stream: io.BytesIO) -> str:
    """Extracts text from DOCX stream using python-docx.

    Args:
        stream: Byte stream of the DOCX file.

    Returns:
        str: Joined text extracted from document paragraphs.

    Raises:
        RuntimeError: If python-docx is not installed.
    """
    if docx is None:
        raise RuntimeError("python-docx is not installed. Unable to process DOCX files.")
    doc = docx.Document(stream)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_file(file_storage: Any) -> str:
    """Extracts plain text from an uploaded file storage object.

    Args:
        file_storage: File-like or Werkzeug FileStorage object to extract text from.

    Returns:
        str: Extracted document text.

    Raises:
        RuntimeError: If pypdf or python-docx is required but not installed.
    """
    filename = getattr(file_storage, "filename", "") or ""
    ext = os.path.splitext(filename)[1].lower()
    stream = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    if ext == ".pdf":
        return _extract_pdf_pages(stream)
    if ext == ".docx":
        return _extract_docx_paragraphs(stream)
    raw_bytes = stream.getvalue()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


# ============================================================================
# SHARED GEMINI API INTEGRATION
# ============================================================================

def _is_invalid_key(api_key: Optional[str]) -> bool:
    """Helper to check whether a Gemini API key is missing or placeholder.

    Args:
        api_key: API key string to test.

    Returns:
        bool: True if key is empty or placeholder, False otherwise.

    Raises:
        None.
    """
    return not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here"


def _send_gemini_request(
    payload: Dict[str, Any], api_key: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> Optional[requests.Response]:
    """Helper to dispatch HTTP POST request to Gemini API.

    Args:
        payload: Dict payload sent to the API.
        api_key: Google Gemini API key.
        timeout: Request timeout in seconds. Defaults to 30.

    Returns:
        Optional[requests.Response]: Response object if request succeeds, None on error.

    Raises:
        None.
    """
    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_API_URL}?key={api_key}"
    try:
        return requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception:
        return None


def _parse_candidate_text(response: requests.Response) -> Optional[str]:
    """Helper to parse first candidate output text from Gemini response.

    Args:
        response: Response object returned from Gemini API call.

    Returns:
        Optional[str]: Clean candidate text if found, None otherwise.

    Raises:
        None.
    """
    if response.status_code != HTTP_STATUS_OK:
        return None
    try:
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except Exception:
        pass
    return None


def _build_gemini_json_payload(
    prompt: str,
    system_instruction: Optional[str],
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """Builds payload for JSON mode Gemini request.

    Args:
        prompt: User prompt text.
        system_instruction: Optional system instruction.
        temperature: Sampling temperature for model response.
        max_output_tokens: Token generation limit.

    Returns:
        Dict[str, Any]: Formatted request payload.

    Raises:
        None.
    """
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": temperature,
            "maxOutputTokens": min(max_output_tokens, 4096),
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return payload


def _build_gemini_text_payload(
    prompt: str, system_instruction: Optional[str]
) -> Dict[str, Any]:
    """Builds payload for text generation Gemini request.

    Args:
        prompt: User prompt text.
        system_instruction: Optional system instruction.

    Returns:
        Dict[str, Any]: Formatted request payload.

    Raises:
        None.
    """
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": DEFAULT_TEMPERATURE,
            "maxOutputTokens": DEFAULT_TEXT_MAX_TOKENS,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return payload


def _parse_json_markdown(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses JSON from markdown code block or raw text.

    Args:
        raw_text: String response possibly wrapped in markdown fence.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary, or None on error.

    Raises:
        None.
    """
    text_clean = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    text_clean = re.sub(r"\s*```$", "", text_clean)
    try:
        return json.loads(text_clean)
    except Exception:
        return None


def call_gemini_api(
    prompt: str,
    api_key: str,
    system_instruction: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Optional[Dict[str, Any]]:
    """Calls Google Gemini API with prompt and returns parsed JSON response.

    Args:
        prompt: User prompt sent to the Gemini model.
        api_key: Google Gemini API key.
        system_instruction: Optional system instruction for the model.
        temperature: Sampling temperature (defaults to 0.2).
        max_output_tokens: Maximum tokens in generationConfig (capped at 4096).

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary, or None if failed.

    Raises:
        None.
    """
    if _is_invalid_key(api_key):
        return None
    payload = _build_gemini_json_payload(
        prompt, system_instruction, temperature=temperature, max_output_tokens=max_output_tokens
    )
    resp = _send_gemini_request(payload, api_key, timeout=DEFAULT_TIMEOUT_SECONDS)
    if not resp:
        return None
    raw_text = _parse_candidate_text(resp)
    return _parse_json_markdown(raw_text) if raw_text else None


def call_gemini_text(
    prompt: str, api_key: str, system_instruction: Optional[str] = None
) -> Optional[str]:
    """Calls Gemini API directly and returns an unstructured text response.

    Args:
        prompt: User prompt containing document context and question.
        api_key: Google Gemini API key.
        system_instruction: Optional system instruction for grounding.

    Returns:
        Optional[str]: Response text from Gemini, or None if unavailable.

    Raises:
        None.
    """
    if _is_invalid_key(api_key):
        return None
    payload = _build_gemini_text_payload(prompt, system_instruction)
    resp = _send_gemini_request(payload, api_key, timeout=DEFAULT_TIMEOUT_SECONDS)
    if not resp:
        return None
    text_out = _parse_candidate_text(resp)
    return text_out.strip() if text_out else None


# ============================================================================
# SAMPLE CONTRACT DATA
# ============================================================================

SAMPLE_CONTRACTS: Dict[str, Dict[str, str]] = {
    "vendor_sla": {
        "title": "Vendor Master Services & SLA (High Risk Terms)",
        "filename": "Vendor_Master_Services_Agreement.txt",
        "content": """MASTER SERVICES AGREEMENT

1. SERVICES AND DELIVERABLES
Vendor shall provide cloud infrastructure and maintenance services as outlined in Statement of Work #1. Client shall provide all necessary access and cooperation in a timely manner.

2. PAYMENT TERMS
Client shall pay all undisputed invoices within fifteen (15) days of receipt. Late payments shall incur a penalty fee of 5% per month or the maximum rate permitted by law. In addition, Client shall forfeit any SLA credits if an invoice is overdue by more than seven (7) days.

3. TERM AND AUTOMATIC RENEWAL
This Agreement shall commence on the Effective Date and continue for an initial term of three (3) years. Thereafter, this Agreement shall automatically renew for successive two (2) year terms, unless Client provides written notice of non-renewal at least ninety (90) days prior to the expiration of the then-current term.

4. TERMINATION
Vendor may terminate this Agreement immediately without cause upon written notice at its sole discretion. Client may terminate only upon sixty (60) days written notice in the event of a material breach that remains uncured for forty-five (45) days.

5. INDEMNIFICATION AND HOLD HARMLESS
Client shall defend, indemnify, and hold harmless Vendor, its affiliates, officers, directors, and contractors from and against any and all claims, liabilities, losses, damages, and expenses (including attorney's fees) arising out of or related to Client's use of the Services or any data provided by Client. Vendor provides no indemnification to Client.

6. LIMITATION OF LIABILITY
VENDOR'S TOTAL AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL BE STRICTLY LIMITED TO $100. UNDER NO CIRCUMSTANCES SHALL VENDOR BE LIABLE FOR ANY CONSEQUENTIAL, INDIRECT, SPECIAL, OR PUNITIVE DAMAGES. CLIENT'S LIABILITY UNDER THIS AGREEMENT SHALL BE UNLIMITED AND SHALL NOT BE SUBJECT TO ANY CAP.

7. INTELLECTUAL PROPERTY RIGHTS
All work product, custom configurations, software improvements, and derivatives created during the term of this Agreement shall be the sole and exclusive property of Vendor, regardless of Client's contribution.

8. CONFIDENTIALITY
Client shall maintain in strict confidence all proprietary pricing, architecture, and documentation of Vendor for a period of five (5) years. Vendor shall use commercially reasonable efforts to protect Client data but assumes no liability for unauthorized third-party breaches.

9. DISPUTE RESOLUTION AND MANDATORY ARBITRATION
All disputes arising out of this Agreement shall be resolved through binding individual arbitration administered by AAA in Wilmington, Delaware. CLIENT EXPRESSLY WAIVES ANY RIGHT TO A TRIAL BY JURY AND WAIVES ANY RIGHT TO PARTICIPATE IN A CLASS ACTION OR REPRESENTATIVE LAWSUIT.

10. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles.
"""
    },
    "mutual_nda": {
        "title": "Mutual Non-Disclosure Agreement (Standard Low-Risk)",
        "filename": "Mutual_Non_Disclosure_Agreement.txt",
        "content": """MUTUAL NON-DISCLOSURE AGREEMENT

1. PURPOSE
The parties wish to explore a potential business relationship and, in connection with this purpose, each party may disclose to the other confidential information.

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public proprietary information disclosed by one party ("Disclosing Party") to the other party ("Receiving Party") that is marked as confidential or that reasonably should be understood to be confidential.

3. EXCLUSIONS FROM CONFIDENTIALITY
Confidential Information does not include information that: (a) is or becomes publicly known through no breach of this Agreement; (b) was already known to Receiving Party without restriction; (c) is independently developed without reference to the Disclosing Party's information; or (d) is received from a third party without duty of confidentiality.

4. OBLIGATIONS OF RECEIVING PARTY
Receiving Party agrees to: (a) protect Disclosing Party's Confidential Information with the same degree of care it uses for its own confidential information, but not less than reasonable care; (b) use Confidential Information solely for the Purpose; and (c) restrict disclosure only to employees and advisers with a need to know.

5. PERMITTED DISCLOSURES
Receiving Party may disclose Confidential Information to the extent required by applicable law or court order, provided Receiving Party gives prompt written notice to Disclosing Party to allow Disclosing Party to seek a protective order.

6. TERM AND TERMINATION
This Agreement shall remain in effect for a term of two (2) years from the Effective Date. The confidentiality obligations herein shall survive termination for a period of three (3) years.

7. RETURN OF MATERIALS
Upon written request of Disclosing Party or upon termination, Receiving Party shall promptly return or certify the destruction of all Confidential Information, provided that backup copies may be retained in compliance with archival policy.

8. REMEDIES
The parties acknowledge that an unauthorized disclosure of Confidential Information may cause irreparable harm for which monetary damages alone would be inadequate, and each party may seek injunctive relief in addition to any legal remedies.

9. GOVERNING LAW AND JURISDICTION
This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without giving effect to conflicts of laws principles. Any legal action shall be instituted in the state or federal courts located in New York County.
"""
    },
    "employment_ip": {
        "title": "Employment & IP Agreement (Modified Version for Comparison)",
        "filename": "Employment_IP_Agreement_V2.txt",
        "content": """PROPRIETARY INFORMATION AND INVENTIONS AGREEMENT

1. EMPLOYMENT RELATIONSHIP
Employee agrees to devote full business time and best efforts to the performance of duties assigned by Company.

2. PROPRIETARY INFORMATION OBLIGATIONS
Employee agrees to hold Company Proprietary Information in strict confidence during and after employment, and shall not disclose or use such information except as required in the course of employment.

3. ASSIGNMENT OF INVENTIONS
Employee hereby assigns to Company all right, title, and interest in and to any inventions, designs, software, or copyrightable works created during employment that relate to Company's business or result from tasks performed for Company.

4. RESTRICTIVE COVENANTS AND NON-COMPETE
During employment and for a period of twelve (12) months following termination, Employee shall not directly or indirectly engage in, advise, or consult for any business that competes with Company within a 50-mile radius of any Company office.

5. NON-SOLICITATION OF CLIENTS AND EMPLOYEES
For a period of twelve (12) months following termination, Employee shall not solicit or encourage any Company employee or contractor to terminate their relationship, nor solicit any customer of Company.

6. TERMINATION
Employment is at-will. Either party may terminate employment at any time with two (2) weeks written notice, or immediately for cause by Company.

7. GOVERNING LAW AND ARBITRATION
This Agreement is governed by the laws of California. Any disputes shall be submitted to confidential arbitration before JAMS in San Francisco, California.
"""
    }
}
