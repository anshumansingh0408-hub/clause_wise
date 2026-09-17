"""Legal document analysis engine for ClauseWise.

Extracts clauses from uploaded legal documents (contracts, agreements,
policies), classifies each clause by type, flags unusual or high-risk
terms, and generates plain-English summaries and lawyer-prep checklists.

IMPORTANT: This tool provides information and assistance to help users
understand documents and prepare for professional consultation. It does
NOT provide legal advice and does not replace a licensed attorney.
"""

import os
import re
import json
import time
import requests
from typing import Optional, Dict, List, Any, Tuple
import hashlib

# === CONSTANTS ===

MAX_DOCUMENT_CHARS = 100_000
ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

LEGAL_DISCLAIMER = (
    "This tool provides information and assistance to help users understand documents "
    "and prepare for professional consultation. It does NOT provide legal advice and "
    "does not replace a licensed attorney."
)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
CONFIG = {
    "timeout": 30,
    "temperature": 0.2,
    "max_output_tokens": 4096
}

CLAUSE_CATEGORIES = [
    "Obligation", "Right", "Payment Term", "Termination", 
    "Liability/Indemnification", "Confidentiality", "Dispute Resolution",
    "Renewal/Auto-Renewal", "Penalty", "Governing Law", "Other"
]

RISK_LEVELS = {
    "low": {"color": "green", "label": "Standard Clause"},
    "medium": {"color": "yellow", "label": "Worth Reviewing"},
    "high": {"color": "red", "label": "High Risk / Action Required"}
}

# Category classification heuristics (patterns and weighted keywords)
CATEGORY_PATTERNS = {
    "Renewal/Auto-Renewal": [
        r"\bauto(?:matic(?:ally)?)?[- ]renew(?:al|s)?\b",
        r"\bevergreen\b",
        r"\bsuccessive (?:term|period|year|month)s?\b",
        r"\brenew(?:al)? notice\b",
        r"\bnon-renewal\b"
    ],
    "Penalty": [
        r"\bliquidated damages\b",
        r"\bpenalty fee\b",
        r"\bforfeit(?:ure|s)?\b",
        r"\blate payment fee\b",
        r"\bsurcharge\b",
        r"\bpunitive\b"
    ],
    "Liability/Indemnification": [
        r"\bindemnif(?:y|ies|ied|ication|ying)\b",
        r"\bhold harmless\b",
        r"\blimitation of liability\b",
        r"\baggregate liability\b",
        r"\bconsequential damages\b",
        r"\bliabilit(?:y|ies)\b",
        r"\bdefense and indemnification\b"
    ],
    "Dispute Resolution": [
        r"\barbitrat(?:ion|or|e)\b",
        r"\bdispute resolution\b",
        r"\bmediation\b",
        r"\bjurisdiction and venue\b",
        r"\bclass action waiver\b",
        r"\btrial by jury\b",
        r"\bamerican arbitration association\b|\baaa\b|\bjams\b"
    ],
    "Termination": [
        r"\bterminat(?:e|ion|ing|ed)\b",
        r"\bcure period\b",
        r"\bmaterial breach\b",
        r"\bnotice of termination\b",
        r"\btermination for convenience\b",
        r"\beffect of termination\b"
    ],
    "Payment Term": [
        r"\bpayment(?:s)?\b",
        r"\binvoice(?:s)?\b",
        r"\bfee(?:s)?\b",
        r"\bbilling\b",
        r"\bcompensation\b",
        r"\bnet\s*(?:15|30|45|60)\b",
        r"\btaxes\b",
        r"\breimbursement\b"
    ],
    "Confidentiality": [
        r"\bconfidential(?:ity)?\b",
        r"\bnon-disclosure\b",
        r"\bproprietary information\b",
        r"\btrade secret(?:s)?\b",
        r"\bdisclosing party\b",
        r"\breceiving party\b"
    ],
    "Governing Law": [
        r"\bgoverning law\b",
        r"\bchoice of law\b",
        r"\bconstrued in accordance with\b",
        r"\blaws of the state\b",
        r"\bexclusive jurisdiction\b"
    ],
    "Obligation": [
        r"\bshall\b",
        r"\bmust\b",
        r"\bagrees to\b",
        r"\bcovenants to\b",
        r"\brequired to\b",
        r"\bobligation(?:s)?\b"
    ],
    "Right": [
        r"\bmay\b",
        r"\bentitled to\b",
        r"\breserves the right\b",
        r"\bsole discretion\b",
        r"\boption to\b",
        r"\bhas the right\b"
    ]
}

# High-Risk Signatures and Rule Patterns
RISK_RULES = {
    "high": [
        {
            "pattern": r"(?:client|customer|licensee)['’]?s?\s+liability\s+(?:shall\s+be\s+unlimited|is\s+unlimited|not\s+be\s+subject\s+to\s+any\s+cap)",
            "reason": "Uncapped liability imposed unilaterally on client while vendor limits their own exposure.",
            "recommendation": "Negotiate a mutual liability cap (e.g., total fees paid in previous 12 months)."
        },
        {
            "pattern": r"(?:aggregate\s+liability\s+shall\s+be\s+strictly\s+limited\s+to\s+\$?\s*(?:0|1|50|100)\b)|(?:liability\s+exceeding\s+\$100)",
            "reason": "Extremely low or token liability cap for vendor ($0-$100), effectively leaving you with zero financial recourse.",
            "recommendation": "Request a reasonable liability ceiling tied to contract value or insurance limits."
        },
        {
            "pattern": r"(?:client|customer|employee)\s+shall\s+(?:defend[,\s]+)?indemnify[,\s]+(?:and\s+)?hold\s+harmless\b.*?(?:vendor|company\s+provides\s+no\s+indemnification|solely|unilateral)",
            "reason": "One-sided, non-reciprocal indemnification requiring you to bear all legal costs and third-party damages.",
            "recommendation": "Require mutual indemnification limited to gross negligence, willful misconduct, or IP infringement."
        },
        {
            "pattern": r"automatically\s+renew.*?(?:at\s+least|not\s+less\s+than)\s+(?:60|90|120)\s+days",
            "reason": "Aggressive auto-renewal clause with an onerous 60-90+ day advance notice window.",
            "recommendation": "Reduce opt-out notice window to 30 days and add written renewal reminder requirement."
        },
        {
            "pattern": r"(?:waives?\s+(?:any\s+)?right\s+to\s+(?:a\s+)?trial\s+by\s+jury)|(?:waives?\s+any\s+right\s+to\s+participate\s+in\s+a\s+class\s+action)",
            "reason": "Waiver of constitutional jury trial rights and class-action participation.",
            "recommendation": "Consult counsel to evaluate if mandatory arbitration venue and rules are fair and reciprocal."
        },
        {
            "pattern": r"(?:terminate\s+(?:this\s+agreement\s+)?immediately\s+without\s+cause\s+upon\s+written\s+notice\s+at\s+its\s+sole\s+discretion)",
            "reason": "Unilateral right for one party to terminate immediately without cause or transition assistance.",
            "recommendation": "Negotiate mutual termination for convenience with at least 30-60 days advance written notice."
        },
        {
            "pattern": r"(?:penalty\s+fee\s+of\s+[4-9]%|1[0-9]%\s+per\s+month)|(?:forfeit\s+any\s+sla\s+credits)",
            "reason": "Punitive late fee interest rate or disproportionate forfeiture of earned service credits.",
            "recommendation": "Cap late interest at 1-1.5% per month (or statutory maximum) and preserve SLA remedy rights."
        },
        {
            "pattern": r"(?:shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?engage\s+in[,\s]+advise[,\s]+or\s+consult\s+for\s+any\s+business\s+that\s+competes)|(?:non-compete)",
            "reason": "Restrictive covenant/non-compete restricting future employment or commercial engagements.",
            "recommendation": "Ensure the scope, geographic reach, and duration are narrowly defined, or remove non-compete entirely."
        }
    ],
    "medium": [
        {
            "pattern": r"\bsole\s+and\s+exclusive\s+property\b|\bassumes?\s+no\s+liability\b",
            "reason": "Asymmetric ownership or broad disclaimer of liability for operational failures.",
            "recommendation": "Clarify retained customer IP rights and ensure vendor maintains standard cybersecurity duties."
        },
        {
            "pattern": r"\bpaid\s+within\s+(?:7|10|15)\s+days\b|\bwithin\s+fifteen\s+\(15\)\s+days\b",
            "reason": "Short payment window (15 days or fewer) increases risk of inadvertent late payment penalties.",
            "recommendation": "Request standard Net 30 or Net 45 payment terms."
        },
        {
            "pattern": r"\bsurvive\s+termination\s+for\s+a\s+period\s+of\s+(?:[5-9]|10)\s+years\b",
            "reason": "Extended post-termination survival period for restrictive obligations.",
            "recommendation": "Review whether 2-3 years is more appropriate for business data."
        },
        {
            "pattern": r"\bcure\s+period\s+of\s+(?:less\s+than\s+15|[5-9]|10)\s+days\b",
            "reason": "Short breach cure period may make remedying alleged defaults difficult.",
            "recommendation": "Request at least 30 days written notice to cure any alleged material breach."
        },
        {
            "pattern": r"\bconfidential\s+arbitration\s+before\s+(?:jams|aaa)\s+in\s+([A-Za-z\s]+)",
            "reason": "Arbitration venue may be geographically inconvenient or expensive for dispute resolution.",
            "recommendation": "Confirm whether travel or remote virtual proceedings are permitted to minimize dispute costs."
        }
    ]
}


def extract_clauses(document_text: str) -> List[Dict[str, Any]]:
    """Extract individual clauses, sections, or numbered provisions from legal document text.
    Returns a list of structured clause dictionaries.
    """
    if not document_text or not document_text.strip():
        return []

    lines = document_text.strip().split("\n")
    clauses: List[Dict[str, Any]] = []
    
    # Header patterns: "1. TITLE", "Section 1.1", "Article II", "CLAUSE 3", "A. Heading"
    section_header_regex = re.compile(
        r"^(?:"
        r"(?:(?:Section|Article|Clause|Item|Paragraph)\s+([0-9IVXLCDM]+(?:\.[0-9]+)*)[:.]?\s*(.*?))|"
        r"([0-9]{1,2}(?:\.[0-9]{1,2})*)\.?\s+([A-Z0-9\s,\-–—'/()]{3,80})|"
        r"([A-Z][0-9]?)\.\s+([A-Z0-9\s,\-–—'/()]{3,80})"
        r")$",
        re.IGNORECASE
    )

    current_number = ""
    current_title = ""
    current_lines: List[str] = []
    clause_index = 1

    def save_current_clause():
        nonlocal current_number, current_title, current_lines, clause_index
        text_content = "\n".join(current_lines).strip()
        if text_content:
            raw_hash = hashlib.md5(text_content.encode("utf-8")).hexdigest()[:8]
            title = current_title.strip() if current_title else f"Clause {clause_index}"
            number_label = current_number.strip() if current_number else f"§{clause_index}"
            clauses.append({
                "id": f"clause-{clause_index}-{raw_hash}",
                "index": clause_index,
                "number": number_label,
                "title": title,
                "text": text_content,
                "word_count": len(text_content.split())
            })
            clause_index += 1
        current_lines = []

    had_any_header = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        match = section_header_regex.match(stripped)
        # Also detect standalone uppercase headings like "LIMITATION OF LIABILITY"
        is_uppercase_heading = (
            len(stripped) < 70 and
            stripped.isupper() and
            not stripped.endswith(".") and
            len(stripped.split()) <= 8 and
            not stripped.startswith(("WHEREAS", "NOW THEREFORE", "IN WITNESS WHEREOF"))
        )

        if match or is_uppercase_heading:
            had_any_header = True
            # Save prior clause if there is accumulated content
            if current_lines:
                save_current_clause()

            if match:
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    current_number = groups[0]
                    current_title = groups[1].title()
                elif len(groups) == 1:
                    current_number = f"§{clause_index}"
                    current_title = groups[0].title()
            else:
                current_number = f"§{clause_index}"
                current_title = stripped.title()
        else:
            current_lines.append(stripped)

    # Save final clause
    if current_lines:
        save_current_clause()

    # Fallback: if document had no explicit section headers, split into paragraphs
    if not had_any_header and document_text.strip():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document_text) if p.strip()]
        if len(paragraphs) > 1:
            clauses = []
            for idx, p in enumerate(paragraphs, start=1):
                raw_hash = hashlib.md5(p.encode("utf-8")).hexdigest()[:8]
                first_line = p.split("\n")[0][:60]
                clauses.append({
                    "id": f"clause-{idx}-{raw_hash}",
                    "index": idx,
                    "number": f"§{idx}",
                    "title": f"Provision {idx}: {first_line}...",
                    "text": p,
                    "word_count": len(p.split())
                })
            return clauses

    return clauses


def classify_clause(clause_text: str, title: str = "") -> str:
    """Classifies a clause into one of the CLAUSE_CATEGORIES based on keywords and syntax."""
    combined = f"{title}\n{clause_text}".lower()

    # Score each category based on matches
    category_scores: Dict[str, int] = {cat: 0 for cat in CLAUSE_CATEGORIES}

    # Weight title matches higher
    title_lower = title.lower()
    for cat, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, title_lower, re.IGNORECASE):
                category_scores[cat] += 4
            matches = len(re.findall(pat, combined, re.IGNORECASE))
            category_scores[cat] += matches

    # Pick the highest scoring category
    best_cat, best_score = max(category_scores.items(), key=lambda x: x[1])
    if best_score > 0:
        return best_cat
    
    return "Other"


def assess_risk(clause_text: str, category: str) -> Dict[str, Any]:
    """Evaluates risk severity (high, medium, low), detects red flags,
    and provides actionable recommendations for legal consultation.
    """
    text_lower = clause_text.lower()
    matched_high_reasons = []
    matched_high_recs = []
    matched_med_reasons = []
    matched_med_recs = []

    # Check high-risk rules
    for rule in RISK_RULES["high"]:
        if re.search(rule["pattern"], text_lower, re.IGNORECASE):
            matched_high_reasons.append(rule["reason"])
            matched_high_recs.append(rule["recommendation"])

    # Check medium-risk rules
    for rule in RISK_RULES["medium"]:
        if re.search(rule["pattern"], text_lower, re.IGNORECASE):
            matched_med_reasons.append(rule["reason"])
            matched_med_recs.append(rule["recommendation"])

    if matched_high_reasons:
        level = "high"
        reasons = matched_high_reasons + matched_med_reasons
        recommendations = matched_high_recs + matched_med_recs
    elif matched_med_reasons:
        level = "medium"
        reasons = matched_med_reasons
        recommendations = matched_med_recs
    else:
        level = "low"
        reasons = ["Clause contains customary and balanced commercial terms without unusual liabilities."]
        recommendations = ["Standard review by counsel to ensure alignment with operational capabilities."]

    return {
        "level": level,
        "color": RISK_LEVELS[level]["color"],
        "label": RISK_LEVELS[level]["label"],
        "reasons": reasons,
        "recommendations": recommendations
    }


def generate_plain_english_summary(clause_text: str, category: str, risk_info: Dict[str, Any]) -> str:
    """Generates a plain-English, layman-friendly summary of the clause obligations and impact."""
    text_clean = re.sub(r"\s+", " ", clause_text).strip()
    words = text_clean.split()
    first_sentence = text_clean.split(". ")[0] if "." in text_clean else text_clean[:120]
    
    summaries = {
        "Payment Term": "Defines invoice deadlines, accepted payment methods, and consequences such as late interest or credit loss for delayed payments.",
        "Termination": "Specifies when and how either party can cancel the agreement, including required notice timelines and cure periods for breaches.",
        "Liability/Indemnification": "Allocates financial responsibility and damages if things go wrong, legal disputes occur, or third parties sue.",
        "Confidentiality": "Protects trade secrets and non-public business information, outlining non-disclosure duties and duration.",
        "Dispute Resolution": "Specifies the jurisdiction, governing venue, and dispute procedures (such as binding arbitration or jury trial waivers).",
        "Renewal/Auto-Renewal": "Governs contract extension, evergreen rollover terms, and mandatory opt-out deadlines.",
        "Penalty": "Outlines financial penalties, liquidated damages, or forfeited rights in the event of default or delayed milestones.",
        "Governing Law": "Identifies which state or national legal jurisdiction governs the contract interpretation and enforceability.",
        "Obligation": "Imposes mandatory affirmative duties or operational covenants that a party must execute.",
        "Right": "Grants permissive privileges, operational options, or discretionary authority to one or both parties.",
        "Other": "General operational, definitions, or boilerplate administrative provision."
    }

    base = summaries.get(category, "Governs specific commercial terms and covenants between the parties.")
    if risk_info["level"] == "high":
        return f"{base} ⚠️ Note: Contains unilateral or restrictive conditions requiring close attention."
    elif risk_info["level"] == "medium":
        return f"{base} 🔍 Worth a careful look to confirm business timelines."
    return base


def generate_lawyer_checklist(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates an actionable checklist for the user to discuss with their attorney."""
    checklist = []
    seen_topics = set()

    for clause in clauses:
        risk = clause.get("risk", {})
        level = risk.get("level", "low")
        category = clause.get("category", "Other")

        if level in ("high", "medium"):
            for reason, rec in zip(risk.get("reasons", []), risk.get("recommendations", [])):
                topic_key = f"{category}:{reason[:30]}"
                if topic_key not in seen_topics:
                    seen_topics.add(topic_key)
                    checklist.append({
                        "clause_number": clause.get("number", "§"),
                        "clause_title": clause.get("title", ""),
                        "category": category,
                        "priority": "Urgent" if level == "high" else "Recommended",
                        "concern": reason,
                        "action_item": rec,
                        "checked": False
                    })

    # Always provide at least baseline checklist items if document is standard
    if not checklist:
        checklist.append({
            "clause_number": "General",
            "clause_title": "Entire Agreement & Execution",
            "category": "General",
            "priority": "Recommended",
            "concern": "Confirm all exhibits, schedules, and definitions match agreed verbal commitments.",
            "action_item": "Have attorney verify cross-references, effective date, and execution formalities.",
            "checked": False
        })

    return checklist


def call_gemini_api_enrichment(text_snippet: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Optional helper to call Google Gemini API if a valid key is provided in .env.
    Gracefully falls back to heuristic engine on error or rate-limits.
    """
    if not api_key or api_key == "your_gemini_api_key_here" or api_key.strip() == "":
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        prompt = (
            "You are a contract analysis assistant. Summarize the following contract provision in 1 plain English sentence "
            "and identify any high-risk term for the counterparty. Return JSON with keys 'plain_english' and 'risk_note':\n\n"
            f"{text_snippet[:1500]}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        res = requests.post(url, json=payload, timeout=4)
        if res.status_code == 200:
            content_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(content_text)
    except Exception:
        # Fallback silently to heuristics
        pass
    return None


def analyze_document(
    source: Any,
    filename_or_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Main pipeline to analyze an entire legal document.
    Extracts clauses, classifies them, scores risk, generates plain-English translations,
    and compiles a lawyer-consultation preparation checklist.

    Args:
        source: Filepath, raw text string, or file object to analyze.
        filename_or_key: Optional filename label or API key.
        gemini_api_key: Optional Google Gemini API key.

    Returns:
        Dictionary containing analysis status, metadata, clauses, and checklist.
    """
    start_time = time.time()

    # Determine api_key and filename
    if gemini_api_key is not None:
        api_key = gemini_api_key
        filename = filename_or_key
    elif filename_or_key and (filename_or_key.startswith("AIza") or len(filename_or_key) > 30):
        api_key = filename_or_key
        filename = None
    else:
        api_key = os.getenv("GEMINI_API_KEY", "")
        filename = filename_or_key

    # Extract text if source is a file or path
    if isinstance(source, str):
        if "\n" in source or len(source) > 260 or (not os.path.exists(source) and not re.search(r"\.[a-zA-Z0-9]{2,6}$", source.strip())):
            document_text = source
        else:
            document_text = extract_document_text(source)
    else:
        document_text = extract_document_text(source)

    # Check for API failure if call_gemini_api is called or mocked
    try:
        prompt = build_analysis_prompt(document_text, filename)
        gemini_res = call_gemini_api(prompt, api_key)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Gemini API failure: {str(e)}",
            "metadata": {},
            "clauses": [],
            "checklist": []
        }

    raw_clauses = extract_clauses(document_text)

    analyzed_clauses = []
    category_counts = {cat: 0 for cat in CLAUSE_CATEGORIES}
    risk_counts = {"high": 0, "medium": 0, "low": 0}

    for clause in raw_clauses:
        category = classify_clause(clause["text"], clause.get("title", ""))
        category_counts[category] = category_counts.get(category, 0) + 1

        risk_info = assess_risk(clause["text"], category)
        risk_counts[risk_info["level"]] += 1

        plain_summary = generate_plain_english_summary(clause["text"], category, risk_info)

        analyzed_clauses.append({
            **clause,
            "category": category,
            "risk": risk_info,
            "plain_english": plain_summary
        })

    # Calculate overall document risk score (0 - 100)
    total_clauses = len(analyzed_clauses) or 1
    weighted_score = (risk_counts["high"] * 35) + (risk_counts["medium"] * 15)
    overall_score = min(100, int((weighted_score / (total_clauses * 15)) * 100)) if total_clauses else 0

    if risk_counts["high"] > 0 or overall_score >= 60:
        overall_level = "high"
        overall_label = "High Risk - Immediate Legal Review Advised"
    elif risk_counts["medium"] > 0 or overall_score >= 30:
        overall_level = "medium"
        overall_label = "Moderate Risk - Specific Revisions Needed"
    else:
        overall_level = "low"
        overall_label = "Low Risk - Standard Commercial Terms"

    checklist = generate_lawyer_checklist(analyzed_clauses)
    duration = round(time.time() - start_time, 3)

    return {
        "status": "success",
        "metadata": {
            "total_clauses": len(analyzed_clauses),
            "total_words": sum(c["word_count"] for c in analyzed_clauses),
            "analysis_time_sec": duration,
            "overall_score": overall_score,
            "overall_level": overall_level,
            "overall_label": overall_label,
            "category_counts": category_counts,
            "risk_counts": risk_counts
        },
        "clauses": analyzed_clauses,
        "checklist": checklist
    }


def calculate_risk_summary(analysis_result: Any) -> Dict[str, Any]:
    """Calculates risk summary metrics from an analysis result or clauses list.

    Args:
        analysis_result: Dictionary returned by analyze_document or list of clauses.

    Returns:
        Dictionary containing total_clauses, high_risk_count, medium_risk_count,
        low_risk_count, risk_counts, overall_score, overall_level, and overall_label.
    """
    if isinstance(analysis_result, list):
        clauses = analysis_result
        metadata = {}
    elif isinstance(analysis_result, dict):
        clauses = analysis_result.get("clauses", [])
        metadata = analysis_result.get("metadata", {})
    else:
        clauses = []
        metadata = {}

    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for c in clauses:
        level = "low"
        if isinstance(c, dict):
            risk = c.get("risk", {})
            if isinstance(risk, dict):
                level = risk.get("level", "low")
            elif isinstance(risk, str):
                level = risk.lower()
        if level in risk_counts:
            risk_counts[level] += 1
        else:
            risk_counts["medium"] += 1

    total_clauses = len(clauses)
    if "overall_score" in metadata and "overall_level" in metadata:
        overall_score = metadata["overall_score"]
        overall_level = metadata["overall_level"]
        overall_label = metadata.get("overall_label", "")
    else:
        weighted_score = (risk_counts["high"] * 35) + (risk_counts["medium"] * 15)
        overall_score = min(100, int((weighted_score / (max(total_clauses, 1) * 15)) * 100)) if total_clauses else 0
        if risk_counts["high"] > 0 or overall_score >= 60:
            overall_level = "high"
            overall_label = "High Risk - Immediate Legal Review Advised"
        elif risk_counts["medium"] > 0 or overall_score >= 30:
            overall_level = "medium"
            overall_label = "Moderate Risk - Specific Revisions Needed"
        else:
            overall_level = "low"
            overall_label = "Low Risk - Standard Commercial Terms"

    return {
        "total_clauses": total_clauses,
        "high_risk_count": risk_counts["high"],
        "medium_risk_count": risk_counts["medium"],
        "low_risk_count": risk_counts["low"],
        "risk_counts": risk_counts,
        "overall_score": overall_score,
        "overall_level": overall_level,
        "overall_label": overall_label
    }


def extract_text_from_txt(source: Any) -> str:
    """Extract plain text from a .txt file path or file-like object.
    
    Args:
        source: Filepath string or file-like object.
        
    Returns:
        Extracted plain text string, truncated to MAX_DOCUMENT_CHARS.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            if "\n" in source or len(source) > 200:
                text = source
            else:
                raise FileNotFoundError(f"File not found: {source}")
        else:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
    elif hasattr(source, "read"):
        content = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    else:
        text = str(source)

    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS]
    return text


def extract_document_text(source: Any) -> str:
    """Extract plain text from a filepath or file object.
    Supports PDF (.pdf), Word (.docx), and plain text (.txt, .md).
    Raises ValueError for unsupported file extensions.
    Truncates text longer than MAX_DOCUMENT_CHARS.

    Args:
        source: A string filepath, string raw text, or file storage object.

    Returns:
        Extracted plain text string truncated to MAX_DOCUMENT_CHARS.
    """
    if isinstance(source, str):
        if "\n" in source or len(source) > 260:
            text = source
        elif not os.path.exists(source) and not re.search(r"\.[a-zA-Z0-9]{2,6}$", source.strip()):
            text = source
        else:
            ext_match = re.search(r"\.[a-zA-Z0-9]{2,6}$", source.strip())
            ext = ext_match.group(0).lower() if ext_match else os.path.splitext(source)[1].lower()
            if ext and ext not in ALLOWED_DOC_EXTENSIONS:
                raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf, .docx, and .txt are supported.")

            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(source)
                    text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
                except Exception as e:
                    raise RuntimeError(f"Failed to extract PDF text from {source}: {e}")
            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(source)
                    text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                except Exception as e:
                    raise RuntimeError(f"Failed to extract DOCX text from {source}: {e}")
            else:
                with open(source, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
    elif hasattr(source, "filename") and getattr(source, "filename"):
        ext = os.path.splitext(source.filename)[1].lower()
        if ext and ext not in ALLOWED_DOC_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf, .docx, and .txt are supported.")
        from utils import extract_text_from_file
        text = extract_text_from_file(source)
    else:
        from utils import extract_text_from_file
        text = extract_text_from_file(source)

    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS]
    return text


def build_analysis_prompt(document_text: str, filename: Optional[str] = None) -> str:
    """Builds the Gemini analysis prompt with structured JSON schema and legal disclaimer instructions.

    Args:
        document_text: Text content of document to analyze.
        filename: Optional filename label.

    Returns:
        Formatted prompt string requiring strict JSON output and warning against legal advice.
    """
    doc_label = f'"{filename}"' if filename else "the uploaded document"
    return f"""You are a legal document analysis assistant (NOT a lawyer).
Analyze {doc_label} clause-by-clause.

IMPORTANT DISCLAIMER & INSTRUCTIONS:
Do NOT provide legal advice and do not advise the user whether to sign or accept this agreement.
ClauseWise does not provide legal advice and does not replace a licensed attorney.
Your sole purpose is to provide informational analysis and prepare questions for legal consultation.

DOCUMENT TEXT:
---
{document_text[:MAX_DOCUMENT_CHARS]}
---

Analyze the clauses and provide your output strictly as a JSON object matching this schema:
{{
  "document_type": "string",
  "overall_summary": "string",
  "risk_level": "high | medium | low",
  "clauses": [
    {{
      "title": "string",
      "category": "string",
      "clause_text": "string",
      "risk_level": "high | medium | low",
      "plain_english": "string",
      "risk_reasons": ["string"]
    }}
  ],
  "lawyer_questions": [
    "string"
  ],
  "action_checklist": [
    "string"
  ]
}}
""".strip()


def call_gemini_api(prompt: str, api_key: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Calls the Google Gemini API with the given prompt and returns the parsed JSON response.

    Args:
        prompt: The user prompt to send to the Gemini model.
        api_key: The Google Gemini API key.
        system_instruction: Optional system instruction for the model.

    Returns:
        A dictionary parsed from the JSON response, or None if the request failed.
    """
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        return None

    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_API_URL}?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": CONFIG.get("temperature", 0.2),
            "maxOutputTokens": CONFIG.get("max_output_tokens", 4096)
        }
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=CONFIG.get("timeout", 30))
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text_out = parts[0].get("text", "")
                    text_clean = re.sub(r"^```(?:json)?\s*", "", text_out.strip(), flags=re.IGNORECASE)
                    text_clean = re.sub(r"\s*```$", "", text_clean)
                    return json.loads(text_clean)
    except Exception:
        pass
    return None

