"""Utility functions for file handling, caching, validation, and 
security helpers used across the ClauseWise application.
"""

import hashlib
import io
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


# ============================================================================
# CONSTANTS
# ============================================================================

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10


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


def secure_filename_custom(filename: str) -> str:
    """Sanitizes a filename to prevent directory traversal and unsafe characters.

    Strips path separators, rejects/strips directory traversal patterns (e.g. '../'),
    and retains only alphanumeric characters, dashes, underscores, and dots.

    Args:
        filename: The original filename or path to sanitize.

    Returns:
        str: A sanitized, safe filename containing only allowed characters.

    Raises:
        None.
    """
    if not filename:
        return ""

    # Repeatedly strip directory traversal sequences like ../ and ..\
    sanitized = filename
    while "../" in sanitized or "..\\" in sanitized:
        sanitized = sanitized.replace("../", "").replace("..\\", "")

    # Strip path separators
    sanitized = sanitized.replace("/", "").replace("\\", "")

    # Keep only alphanumeric, dash, underscore, and dot
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "", sanitized)

    # Collapse multiple consecutive dots to prevent traversal
    while ".." in sanitized:
        sanitized = sanitized.replace("..", ".")

    # Strip leading/trailing dots or spaces
    sanitized = sanitized.strip(". ")

    return sanitized


def generate_cache_key(*args: str) -> str:
    """Generates an MD5 hash cache key from any number of string arguments.

    Computes an MD5 digest of the concatenated arguments for caching AI responses
    and avoiding duplicate processing of identical documents.

    Args:
        *args: Variable number of string arguments (e.g. document text, mode).

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
        return size_bytes <= max_mb * 1024 * 1024
    except OSError:
        return False


def format_confidence_badge(score: float) -> dict:
    """Converts a confidence score (0.0-1.0) into display information.

    Args:
        score: Numerical confidence score between 0.0 and 1.0.

    Returns:
        dict: A dictionary containing:
            - 'label' (str): Descriptive confidence level text.
            - 'color' (str): Badge theme color ('green', 'yellow', or 'red').
            - 'percentage' (int): Score expressed as an integer percentage (0-100).

    Raises:
        None.
    """
    percentage = int(round(score * 100))
    if score >= 0.85:
        return {
            "label": "High Confidence",
            "color": "green",
            "percentage": percentage,
        }
    elif score >= 0.6:
        return {
            "label": "Medium Confidence",
            "color": "yellow",
            "percentage": percentage,
        }
    else:
        return {
            "label": "Low Confidence - Review Carefully",
            "color": "red",
            "percentage": percentage,
        }


def format_risk_badge(risk_level: str) -> dict:
    """Converts a risk_level string into display information.

    Args:
        risk_level: Risk level string ("low", "medium", or "high").

    Returns:
        dict: A dictionary containing:
            - 'label' (str): Descriptive risk badge text.
            - 'color' (str): Theme color ('green', 'yellow', or 'red').
            - 'icon' (str): Icon symbol ('✓' or '⚠').

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
    ip: str, tracker: dict, max_requests: int, window_seconds: int
) -> bool:
    """Generic rate limiter that checks and updates timestamps in place.

    Takes a tracker dict mapping IP to a list of timestamps. Removes timestamps
    older than window_seconds, checks if remaining count is below max_requests,
    and appends current timestamp if allowed.

    Args:
        ip: Client IP address or unique requester identifier.
        tracker: Dictionary mapping IP strings to lists of float timestamps (mutated in place).
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
        text: Input string to sanitize (e.g. a chat question about the document).
        max_length: Maximum permitted length in characters.

    Returns:
        str: Sanitized and truncated string.

    Raises:
        None.
    """
    if not text:
        return ""
    # Remove null bytes and control characters (ASCII 0-31 and 127-159)
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    cleaned = cleaned.strip()
    if max_length is not None and max_length >= 0:
        cleaned = cleaned[:max_length]
    return cleaned


def cleanup_old_uploads(upload_folder: str, max_age_seconds: int = 3600) -> int:
    """Deletes files in upload_folder older than max_age_seconds for privacy.

    Safeguards privacy by removing uploaded legal documents after the retention period.
    Handles missing or invalid directories gracefully.

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

    deleted_count = 0
    now = time.time()

    try:
        for entry in os.listdir(upload_folder):
            entry_path = os.path.join(upload_folder, entry)
            if os.path.isfile(entry_path):
                try:
                    file_age = now - os.path.getmtime(entry_path)
                    if file_age > max_age_seconds:
                        os.remove(entry_path)
                        deleted_count += 1
                except OSError:
                    continue
    except OSError:
        return deleted_count

    return deleted_count


def redact_preview(text: str, max_chars: int = 300) -> str:
    """Truncates document text to max_chars for safe preview display, adding '...' if truncated.

    Used to avoid dumping entire sensitive documents into logs or error messages.

    Args:
        text: Document text to preview.
        max_chars: Maximum character count before truncation. Defaults to 300.

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


# ============================================================================
# DOCUMENT PARSING & SAMPLE CONTRACT DATA
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


def extract_text_from_file(file_storage) -> str:
    """Extracts plain text from an uploaded file storage object.

    Supports PDF (.pdf), Word (.docx), and plain text (.txt, .md).

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
    file_storage.seek(0)  # reset pointer

    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed. Unable to process PDF files.")
        reader = PdfReader(stream)
        extracted = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted.append(page_text)
        return "\n\n".join(extracted)

    elif ext == ".docx":
        if docx is None:
            raise RuntimeError("python-docx is not installed. Unable to process DOCX files.")
        doc = docx.Document(stream)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    else:
        # Default to utf-8 text with fallback
        raw_bytes = stream.getvalue()
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")


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
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
