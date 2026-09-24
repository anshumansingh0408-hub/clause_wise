"""Legal document analysis engine for ClauseWise.

Extracts clauses from uploaded legal documents (contracts, agreements,
policies), classifies each clause by type, flags unusual or high-risk
terms, and generates plain-English summaries and lawyer-prep checklists.

IMPORTANT: This tool provides information and assistance to help users
understand documents and prepare for professional consultation. It does
NOT provide legal advice and does not replace a licensed attorney.
"""

import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    call_gemini_api,
    call_gemini_text,
    extract_text_from_file,
    GEMINI_API_URL,
    GEMINI_MODEL,
)


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

CONFIG = {
    "MAX_DOCUMENT_CHARS": 30000,
    "timeout": 30,
    "temperature": 0.2,
    "max_output_tokens": 4096,
}
MAX_DOCUMENT_CHARS = CONFIG["MAX_DOCUMENT_CHARS"]
ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

SCORE_HIGH_WEIGHT = 35
SCORE_MEDIUM_WEIGHT = 15
SCORE_BASE_DIVISOR = 15
SCORE_MAX = 100
RISK_THRESHOLD_HIGH = 60
RISK_THRESHOLD_MEDIUM = 30

HEADING_MAX_WORDS = 8
HEADING_MAX_LEN = 70
MAX_SNIPPET_LEN = 1500
SHORT_TITLE_LEN = 60
HASH_PREFIX_LEN = 8
DEFAULT_DURATION_DECIMALS = 3

LEGAL_DISCLAIMER = (
    "This tool provides information and assistance to help users understand documents "
    "and prepare for professional consultation. It does NOT provide legal advice and "
    "does not replace a licensed attorney."
)

DEFAULT_POSSIBLE_NEXT_STEPS: List[str] = [
    "Negotiate the clause: Request specific revisions, carveouts, or monetary caps with the counterparty.",
    "Accept as-is: Proceed if commercial value and relationship outweigh the identified risks.",
    "Seek clarification before signing: Request written explanation or confirmation for ambiguous provisions.",
    "Walk away from this term: Evaluate whether high-risk or uncapped obligations are non-negotiable dealbreakers.",
]

CLAUSE_CATEGORIES = [
    "Obligation", "Right", "Payment Term", "Termination",
    "Liability/Indemnification", "Confidentiality", "Dispute Resolution",
    "Renewal/Auto-Renewal", "Penalty", "Governing Law", "Other",
]

RISK_LEVELS = {
    "low": {"color": "green", "label": "Standard Clause"},
    "medium": {"color": "yellow", "label": "Worth Reviewing"},
    "high": {"color": "red", "label": "High Risk / Action Required"},
}

CATEGORY_SUMMARIES: Dict[str, str] = {
    "Payment Term": "Defines invoice deadlines, accepted payment methods, and consequences for delayed payments.",
    "Termination": "Specifies when and how either party can cancel agreement, including required notice timelines.",
    "Liability/Indemnification": "Allocates financial responsibility and damages if disputes occur or third parties sue.",
    "Confidentiality": "Protects trade secrets and proprietary data, outlining non-disclosure duties and duration.",
    "Dispute Resolution": "Specifies governing venue and procedures (such as binding arbitration or jury trial waivers).",
    "Renewal/Auto-Renewal": "Governs contract extension, evergreen rollover terms, and mandatory opt-out deadlines.",
    "Penalty": "Outlines financial penalties, liquidated damages, or forfeited rights in event of default.",
    "Governing Law": "Identifies which legal jurisdiction governs contract interpretation and enforceability.",
    "Obligation": "Imposes mandatory affirmative duties or operational covenants that a party must execute.",
    "Right": "Grants permissive privileges, operational options, or discretionary authority.",
    "Other": "General operational, definitions, or boilerplate administrative provision.",
}

CATEGORY_PATTERNS = {
    "Renewal/Auto-Renewal": [
        r"\bauto(?:matic(?:ally)?)?[- ]renew(?:al|s)?\b",
        r"\bevergreen\b",
        r"\bsuccessive (?:term|period|year|month)s?\b",
        r"\brenew(?:al)? notice\b",
        r"\bnon-renewal\b",
    ],
    "Penalty": [
        r"\bliquidated damages\b",
        r"\bpenalty fee\b",
        r"\bforfeit(?:ure|s)?\b",
        r"\blate payment fee\b",
        r"\bsurcharge\b",
        r"\bpunitive\b",
    ],
    "Liability/Indemnification": [
        r"\bindemnif(?:y|ies|ied|ication|ying)\b",
        r"\bhold harmless\b",
        r"\blimitation of liability\b",
        r"\baggregate liability\b",
        r"\bconsequential damages\b",
        r"\bliabilit(?:y|ies)\b",
        r"\bdefense and indemnification\b",
    ],
    "Dispute Resolution": [
        r"\barbitrat(?:ion|or|e)\b",
        r"\bdispute resolution\b",
        r"\bmediation\b",
        r"\bjurisdiction and venue\b",
        r"\bclass action waiver\b",
        r"\btrial by jury\b",
        r"\bamerican arbitration association\b|\baaa\b|\bjams\b",
    ],
    "Termination": [
        r"\bterminat(?:e|ion|ing|ed)\b",
        r"\bcure period\b",
        r"\bmaterial breach\b",
        r"\bnotice of termination\b",
        r"\btermination for convenience\b",
        r"\beffect of termination\b",
    ],
    "Payment Term": [
        r"\bpayment(?:s)?\b",
        r"\binvoice(?:s)?\b",
        r"\bfee(?:s)?\b",
        r"\bbilling\b",
        r"\bcompensation\b",
        r"\bnet\s*(?:15|30|45|60)\b",
        r"\btaxes\b",
        r"\breimbursement\b",
    ],
    "Confidentiality": [
        r"\bconfidential(?:ity)?\b",
        r"\bnon-disclosure\b",
        r"\bproprietary information\b",
        r"\btrade secret(?:s)?\b",
        r"\bdisclosing party\b",
        r"\breceiving party\b",
    ],
    "Governing Law": [
        r"\bgoverning law\b",
        r"\bchoice of law\b",
        r"\bconstrued in accordance with\b",
        r"\blaws of the state\b",
        r"\bexclusive jurisdiction\b",
    ],
    "Obligation": [
        r"\bshall\b",
        r"\bmust\b",
        r"\bagrees to\b",
        r"\bcovenants to\b",
        r"\brequired to\b",
        r"\bobligation(?:s)?\b",
    ],
    "Right": [
        r"\bmay\b",
        r"\bentitled to\b",
        r"\breserves the right\b",
        r"\bsole discretion\b",
        r"\boption to\b",
        r"\bhas the right\b",
    ],
}

RISK_RULES = {
    "high": [
        {
            "pattern": (
                r"(?:client|customer|licensee)['’]?s?\s+liability\s+"
                r"(?:shall\s+be\s+unlimited|is\s+unlimited|not\s+be\s+subject\s+to\s+any\s+cap)"
            ),
            "reason": "Uncapped liability imposed unilaterally on client while vendor limits their own exposure.",
            "recommendation": "Negotiate a mutual liability cap (e.g., total fees paid in previous 12 months).",
        },
        {
            "pattern": (
                r"(?:aggregate\s+liability\s+shall\s+be\s+strictly\s+limited\s+to\s+\$?\s*(?:0|1|50|100)\b)|"
                r"(?:liability\s+exceeding\s+\$100)"
            ),
            "reason": "Extremely low or token liability cap for vendor ($0-$100), effectively leaving you with zero financial recourse.",
            "recommendation": "Request a reasonable liability ceiling tied to contract value or insurance limits.",
        },
        {
            "pattern": (
                r"(?:client|customer|employee)\s+shall\s+(?:defend[,\s]+)?indemnify[,\s]+"
                r"(?:and\s+)?hold\s+harmless\b.*?(?:vendor|company\s+provides\s+no\s+indemnification|solely|unilateral)"
            ),
            "reason": "One-sided, non-reciprocal indemnification requiring you to bear all legal costs and third-party damages.",
            "recommendation": "Require mutual indemnification limited to gross negligence, willful misconduct, or IP infringement.",
        },
        {
            "pattern": r"automatically\s+renew.*?(?:at\s+least|not\s+less\s+than)\s+(?:60|90|120)\s+days",
            "reason": "Aggressive auto-renewal clause with an onerous 60-90+ day advance notice window.",
            "recommendation": "Reduce opt-out notice window to 30 days and add written renewal reminder requirement.",
        },
        {
            "pattern": (
                r"(?:waives?\s+(?:any\s+)?right\s+to\s+(?:a\s+)?trial\s+by\s+jury)|"
                r"(?:waives?\s+any\s+right\s+to\s+participate\s+in\s+a\s+class\s+action)"
            ),
            "reason": "Waiver of constitutional jury trial rights and class-action participation.",
            "recommendation": "Consult counsel to evaluate if mandatory arbitration venue and rules are fair and reciprocal.",
        },
        {
            "pattern": (
                r"(?:terminate\s+(?:this\s+agreement\s+)?immediately\s+without\s+cause\s+"
                r"upon\s+written\s+notice\s+at\s+its\s+sole\s+discretion)"
            ),
            "reason": "Unilateral right for one party to terminate immediately without cause or transition assistance.",
            "recommendation": "Negotiate mutual termination for convenience with at least 30-60 days advance written notice.",
        },
        {
            "pattern": (
                r"(?:penalty\s+fee\s+of\s+[4-9]%|1[0-9]%\s+per\s+month)|"
                r"(?:forfeit\s+any\s+sla\s+credits)"
            ),
            "reason": "Punitive late fee interest rate or disproportionate forfeiture of earned service credits.",
            "recommendation": "Cap late interest at 1-1.5% per month (or statutory maximum) and preserve SLA remedy rights.",
        },
        {
            "pattern": (
                r"(?:shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?engage\s+in[,\s]+advise[,\s]+"
                r"or\s+consult\s+for\s+any\s+business\s+that\s+competes)|(?:non-compete)"
            ),
            "reason": "Restrictive covenant/non-compete restricting future employment or commercial engagements.",
            "recommendation": "Ensure the scope, geographic reach, and duration are narrowly defined, or remove non-compete entirely.",
        },
    ],
    "medium": [
        {
            "pattern": r"\bsole\s+and\s+exclusive\s+property\b|\bassumes?\s+no\s+liability\b",
            "reason": "Asymmetric ownership or broad disclaimer of liability for operational failures.",
            "recommendation": "Clarify retained customer IP rights and ensure vendor maintains standard cybersecurity duties.",
        },
        {
            "pattern": r"\bpaid\s+within\s+(?:7|10|15)\s+days\b|\bwithin\s+fifteen\s+\(15\)\s+days\b",
            "reason": "Short payment window (15 days or fewer) increases risk of inadvertent late payment penalties.",
            "recommendation": "Request standard Net 30 or Net 45 payment terms.",
        },
        {
            "pattern": r"\bsurvive\s+termination\s+for\s+a\s+period\s+of\s+(?:[5-9]|10)\s+years\b",
            "reason": "Extended post-termination survival period for restrictive obligations.",
            "recommendation": "Review whether 2-3 years is more appropriate for business data.",
        },
        {
            "pattern": r"\bcure\s+period\s+of\s+(?:less\s+than\s+15|[5-9]|10)\s+days\b",
            "reason": "Short breach cure period may make remedying alleged defaults difficult.",
            "recommendation": "Request at least 30 days written notice to cure any alleged material breach.",
        },
        {
            "pattern": r"\bconfidential\s+arbitration\s+before\s+(?:jams|aaa)\s+in\s+([A-Za-z\s]+)",
            "reason": "Arbitration venue may be geographically inconvenient or expensive for dispute resolution.",
            "recommendation": "Confirm whether travel or remote virtual proceedings are permitted to minimize dispute costs.",
        },
    ],
}

SECTION_HEADER_REGEX = re.compile(
    r"^(?:"
    r"(?:(?:Section|Article|Clause|Item|Paragraph)\s+([0-9IVXLCDM]+(?:\.[0-9]+)*)[:.]?\s*(.*?))|"
    r"([0-9]{1,2}(?:\.[0-9]{1,2})*)\.?\s+([A-Z0-9\s,\-–—'/()]{3,80})|"
    r"([A-Z][0-9]?)\.\s+([A-Z0-9\s,\-–—'/()]{3,80})"
    r")$",
    re.IGNORECASE,
)


# ============================================================================
# CLAUSE EXTRACTION & PARSING
# ============================================================================

def _build_clause_dict(
    idx: int, num_label: str, title: str, text: str
) -> Dict[str, Any]:
    """Constructs a structured clause dictionary.

    Args:
        idx: 1-indexed sequential clause counter.
        num_label: Display label for clause section number.
        title: Title of the clause.
        text: Full body text of the clause.

    Returns:
        Dict[str, Any]: Structured clause representation.

    Raises:
        None.
    """
    raw_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]
    return {
        "id": f"clause-{idx}-{raw_hash}",
        "index": idx,
        "number": num_label.strip() if num_label else f"§{idx}",
        "title": title.strip() if title else f"Clause {idx}",
        "text": text,
        "word_count": len(text.split()),
    }


def _is_uppercase_heading(stripped: str) -> bool:
    """Checks if a string represents an uppercase standalone heading.

    Args:
        stripped: Stripped text line.

    Returns:
        bool: True if line matches uppercase heading heuristic, False otherwise.

    Raises:
        None.
    """
    return (
        len(stripped) < HEADING_MAX_LEN
        and stripped.isupper()
        and not stripped.endswith(".")
        and len(stripped.split()) <= HEADING_MAX_WORDS
        and not stripped.startswith(("WHEREAS", "NOW THEREFORE", "IN WITNESS WHEREOF"))
    )


def _parse_heading(
    match: Optional[Any], stripped: str, clause_index: int
) -> Tuple[str, str]:
    """Extracts clause number and title from regex match or line.

    Args:
        match: Match object from SECTION_HEADER_REGEX.
        stripped: Stripped header string.
        clause_index: Current clause counter.

    Returns:
        Tuple[str, str]: Tuple of (clause_number, clause_title).

    Raises:
        None.
    """
    if not match:
        return f"§{clause_index}", stripped.title()
    groups = [g for g in match.groups() if g]
    if len(groups) >= 2:
        return groups[0], groups[1].title()
    if len(groups) == 1:
        return f"§{clause_index}", groups[0].title()
    return f"§{clause_index}", stripped.title()


def _split_into_paragraph_clauses(text: str) -> List[Dict[str, Any]]:
    """Splits unformatted text into paragraph clauses.

    Args:
        text: Raw document text without explicit headers.

    Returns:
        List[Dict[str, Any]]: List of paragraph-based clause objects.

    Raises:
        None.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    clauses = []
    for idx, p in enumerate(paragraphs, start=1):
        first_line = p.split("\n")[0][:SHORT_TITLE_LEN]
        title = f"Provision {idx}: {first_line}..."
        clauses.append(_build_clause_dict(idx, f"§{idx}", title, p))
    return clauses


def _append_clause_if_content(
    clauses: List[Dict[str, Any]], num: str, title: str, lines: List[str]
) -> None:
    """Appends clause to list if lines contains non-empty text.

    Args:
        clauses: List of clauses to append to.
        num: Clause number.
        title: Clause title.
        lines: Accumulated line buffer.

    Returns:
        None.
    Raises:
        None.
    """
    if lines:
        clauses.append(_build_clause_dict(len(clauses) + 1, num, title, "\n".join(lines).strip()))


def _process_lines_into_clauses(lines: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    """Scans lines and extracts header-bounded clauses.

    Args:
        lines: List of text lines.

    Returns:
        Tuple[List[Dict[str, Any]], bool]: Extracted clauses and had_header flag.

    Raises:
        None.
    """
    clauses, curr_num, curr_title, curr_lines, had_header = [], "", "", [], False
    for line in lines:
        s = line.strip()
        if not s:
            if curr_lines: curr_lines.append("")
            continue
        m = SECTION_HEADER_REGEX.match(s)
        if m or _is_uppercase_heading(s):
            had_header = True
            _append_clause_if_content(clauses, curr_num, curr_title, curr_lines)
            curr_lines = []
            curr_num, curr_title = _parse_heading(m, s, len(clauses) + 1)
        else:
            curr_lines.append(s)
    _append_clause_if_content(clauses, curr_num, curr_title, curr_lines)
    return clauses, had_header


def extract_clauses(document_text: str) -> List[Dict[str, Any]]:
    """Extract individual clauses, sections, or numbered provisions from legal document text.

    Args:
        document_text: Raw plain text of legal document.

    Returns:
        List[Dict[str, Any]]: Extracted structured clause records.

    Raises:
        None.
    """
    if not document_text or not document_text.strip():
        return []
    lines = document_text.strip().split("\n")
    clauses, had_header = _process_lines_into_clauses(lines)
    if not had_header and len(re.split(r"\n\s*\n", document_text.strip())) > 1:
        return _split_into_paragraph_clauses(document_text)
    return clauses


# ============================================================================
# CLASSIFICATION & RISK ASSESSMENT
# ============================================================================

def classify_clause(clause_text: str, title: str = "") -> str:
    """Classifies a clause into one of CLAUSE_CATEGORIES based on patterns.

    Args:
        clause_text: Body text of clause.
        title: Optional section heading of clause.

    Returns:
        str: Best matching category name or 'Other'.

    Raises:
        None.
    """
    combined = f"{title}\n{clause_text}".lower()
    title_lower = title.lower()
    category_scores = {cat: 0 for cat in CLAUSE_CATEGORIES}
    for cat, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, title_lower, re.IGNORECASE):
                category_scores[cat] += 4
            category_scores[cat] += len(re.findall(pat, combined, re.IGNORECASE))
    best_cat, best_score = max(category_scores.items(), key=lambda x: x[1])
    return best_cat if best_score > 0 else "Other"


def _match_risk_rules(text_lower: str, level: str) -> Tuple[List[str], List[str]]:
    """Evaluates risk rules for a specific severity level.

    Args:
        text_lower: Lowercase clause text.
        level: 'high' or 'medium'.

    Returns:
        Tuple[List[str], List[str]]: Reasons and recommendations matched.

    Raises:
        None.
    """
    reasons, recs = [], []
    for rule in RISK_RULES.get(level, []):
        if re.search(rule["pattern"], text_lower, re.IGNORECASE):
            reasons.append(rule["reason"])
            recs.append(rule["recommendation"])
    return reasons, recs


def _resolve_risk_verdict(
    high_reasons: List[str], high_recs: List[str], med_reasons: List[str], med_recs: List[str]
) -> Tuple[str, List[str], List[str]]:
    """Determines final risk severity level, combined reasons, and recommendations.

    Args:
        high_reasons: Matched high-risk reasons.
        high_recs: Matched high-risk recommendations.
        med_reasons: Matched medium-risk reasons.
        med_recs: Matched medium-risk recommendations.

    Returns:
        Tuple[str, List[str], List[str]]: Level, reasons, recommendations.

    Raises:
        None.
    """
    if high_reasons:
        return "high", high_reasons + med_reasons, high_recs + med_recs
    if med_reasons:
        return "medium", med_reasons, med_recs
    default_reasons = ["Clause contains customary and balanced commercial terms without unusual liabilities."]
    default_recs = ["Standard review by counsel to ensure alignment with operational capabilities."]
    return "low", default_reasons, default_recs


def assess_risk(clause_text: str, category: str) -> Dict[str, Any]:
    """Evaluates risk severity, detects red flags, and provides recommendations.

    Args:
        clause_text: Text of clause to evaluate.
        category: Classified category string.

    Returns:
        Dict[str, Any]: Risk level, color, label, reasons, and recommendations.

    Raises:
        None.
    """
    text_lower = clause_text.lower()
    high_reasons, high_recs = _match_risk_rules(text_lower, "high")
    med_reasons, med_recs = _match_risk_rules(text_lower, "medium")
    level, reasons, recommendations = _resolve_risk_verdict(
        high_reasons, high_recs, med_reasons, med_recs
    )
    return {
        "level": level,
        "color": RISK_LEVELS[level]["color"],
        "label": RISK_LEVELS[level]["label"],
        "reasons": reasons,
        "recommendations": recommendations,
    }


def generate_plain_english_summary(
    clause_text: str, category: str, risk_info: Dict[str, Any]
) -> str:
    """Generates a plain-English, layman-friendly summary of the clause obligations.

    Args:
        clause_text: Original text of the clause.
        category: Clause category string.
        risk_info: Output dictionary from assess_risk.

    Returns:
        str: Plain English description of clause obligations and risk notes.

    Raises:
        None.
    """
    base = CATEGORY_SUMMARIES.get(category, "Governs specific commercial terms between the parties.")
    if risk_info.get("level") == "high":
        return f"{base} ⚠️ Note: Contains unilateral or restrictive conditions requiring close attention."
    if risk_info.get("level") == "medium":
        return f"{base} 🔍 Worth a careful look to confirm business timelines."
    return base


def _make_checklist_item(
    clause: Dict[str, Any], rsn: str, rec: str, lvl: str
) -> Dict[str, Any]:
    """Formats single lawyer checklist item dictionary.

    Args:
        clause: Clause dictionary.
        rsn: Risk concern reason.
        rec: Actionable advice recommendation.
        lvl: Severity level string.

    Returns:
        Dict[str, Any]: Formatted checklist item.

    Raises:
        None.
    """
    return {
        "clause_number": clause.get("number", "§"),
        "clause_title": clause.get("title", ""),
        "category": clause.get("category", "Other"),
        "priority": "Urgent" if lvl == "high" else "Recommended",
        "concern": rsn,
        "action_item": rec,
        "checked": False,
    }


def _extract_urgent_clauses(
    clauses: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Gathers high and medium risk clauses for checklist generation.

    Args:
        clauses: List of analyzed clauses.

    Returns:
        List[Dict[str, Any]]: Formatted checklist questions.

    Raises:
        None.
    """
    items, seen = [], set()
    for clause in clauses:
        risk = clause.get("risk", {})
        lvl, cat = risk.get("level", "low"), clause.get("category", "Other")
        if lvl in ("high", "medium"):
            for rsn, rec in zip(risk.get("reasons", []), risk.get("recommendations", [])):
                key = f"{cat}:{rsn[:30]}"
                if key not in seen:
                    seen.add(key)
                    items.append(_make_checklist_item(clause, rsn, rec, lvl))
    return items


def generate_lawyer_checklist(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates an actionable checklist for the user to discuss with their attorney.

    Args:
        clauses: List of analyzed clause dictionaries.

    Returns:
        List[Dict[str, Any]]: Prioritized checklist items for lawyer consultation.

    Raises:
        None.
    """
    checklist = _extract_urgent_clauses(clauses)
    if not checklist:
        checklist.append({
            "clause_number": "General",
            "clause_title": "Entire Agreement & Execution",
            "category": "General",
            "priority": "Recommended",
            "concern": "Confirm all exhibits, schedules, and definitions match agreed verbal commitments.",
            "action_item": "Have attorney verify cross-references, effective date, and execution formalities.",
            "checked": False,
        })
    return checklist


def call_gemini_api_enrichment(text_snippet: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Calls Gemini API for single clause plain-English enrichment.

    Args:
        text_snippet: Clause text to enrich.
        api_key: Google Gemini API key.

    Returns:
        Optional[Dict[str, Any]]: Enriched JSON or None on failure.

    Raises:
        None.
    """
    prompt = (
        "You are a contract analysis assistant. Summarize the following contract provision in 1 plain English sentence "
        "and identify any high-risk term for the counterparty. Return JSON with keys 'plain_english' and 'risk_note':\n\n"
        f"{text_snippet[:MAX_SNIPPET_LEN]}"
    )
    return call_gemini_api(prompt, api_key)


# ============================================================================
# PIPELINE & DOCUMENT ANALYSIS
# ============================================================================

def _resolve_api_key_and_filename(
    filename_or_key: Optional[str], gemini_api_key: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Resolves Gemini API key and filename from overloaded inputs.

    Args:
        filename_or_key: Filename label or overloaded API key string.
        gemini_api_key: Explicit Gemini API key.

    Returns:
        Tuple[str, Optional[str]]: Tuple of (api_key, filename).

    Raises:
        None.
    """
    if gemini_api_key is not None:
        return gemini_api_key, filename_or_key
    if filename_or_key and (filename_or_key.startswith("AIza") or len(filename_or_key) > 30):
        return filename_or_key, None
    return os.getenv("GEMINI_API_KEY", ""), filename_or_key


def _assemble_clauses(
    raw_clauses: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Enriches extracted clauses with category, risk, and summary.

    Args:
        raw_clauses: List of extracted clause dictionaries.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
            Analyzed clauses, category counts, and risk counts.

    Raises:
        None.
    """
    analyzed = []
    cat_counts = {c: 0 for c in CLAUSE_CATEGORIES}
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    for clause in raw_clauses:
        cat = classify_clause(clause["text"], clause.get("title", ""))
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        r_info = assess_risk(clause["text"], cat)
        risk_counts[r_info["level"]] += 1
        summary = generate_plain_english_summary(clause["text"], cat, r_info)
        analyzed.append({**clause, "category": cat, "risk": r_info, "plain_english": summary})
    return analyzed, cat_counts, risk_counts


def _compute_overall_risk(
    total: int, risk_counts: Dict[str, int]
) -> Tuple[int, str, str]:
    """Calculates overall risk score and label.

    Args:
        total: Total number of clauses.
        risk_counts: Counts of high, medium, and low risk clauses.

    Returns:
        Tuple[int, str, str]: Tuple of (score, level, label).

    Raises:
        None.
    """
    divisor = max(total, 1) * SCORE_BASE_DIVISOR
    weighted = (risk_counts["high"] * SCORE_HIGH_WEIGHT) + (risk_counts["medium"] * SCORE_MEDIUM_WEIGHT)
    score = min(SCORE_MAX, int((weighted / divisor) * SCORE_MAX)) if total else 0
    if risk_counts["high"] > 0 or score >= RISK_THRESHOLD_HIGH:
        return score, "high", "High Risk - Immediate Legal Review Advised"
    if risk_counts["medium"] > 0 or score >= RISK_THRESHOLD_MEDIUM:
        return score, "medium", "Moderate Risk - Specific Revisions Needed"
    return score, "low", "Low Risk - Standard Commercial Terms"


def _extract_next_steps(gemini_res: Any) -> List[str]:
    """Extracts 2-4 generic informational next steps from Gemini response.

    Args:
        gemini_res: Parsed Gemini response object.

    Returns:
        List[str]: List of 2-4 next-step paths or default paths.

    Raises:
        None.
    """
    if isinstance(gemini_res, dict):
        raw = gemini_res.get("possible_next_steps")
        if isinstance(raw, list) and raw:
            cleaned = [str(s).strip() for s in raw if str(s).strip()]
            if 2 <= len(cleaned) <= 6:
                return cleaned[:4]
    return list(DEFAULT_POSSIBLE_NEXT_STEPS)


def _format_analysis_metadata(
    analyzed: List[Dict[str, Any]], risk_counts: Dict[str, int], duration: float, cat_counts: Dict[str, int]
) -> Dict[str, Any]:
    """Builds metadata dictionary for analysis result.

    Args:
        analyzed: List of analyzed clauses.
        risk_counts: Count of risks by level.
        duration: Processing time in seconds.
        cat_counts: Category counts.

    Returns:
        Dict[str, Any]: Metadata dictionary.

    Raises:
        None.
    """
    score, level, label = _compute_overall_risk(len(analyzed), risk_counts)
    words = sum(c["word_count"] for c in analyzed)
    return {
        "total_clauses": len(analyzed), "total_words": words, "analysis_time_sec": duration,
        "overall_score": score, "overall_level": level, "overall_label": label,
        "category_counts": cat_counts, "risk_counts": risk_counts,
    }


def _format_analysis_result(
    analyzed: List[Dict[str, Any]], cat_counts: Dict[str, int], risk_counts: Dict[str, int], start: float, next_steps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Packages final analysis metadata and payload.

    Args:
        analyzed: List of analyzed clauses.
        cat_counts: Category frequencies.
        risk_counts: Risk severity counts.
        start: Start timestamp.
        next_steps: Optional list of informational next-step paths.

    Returns:
        Dict[str, Any]: Success payload.

    Raises:
        None.
    """
    duration = round(time.time() - start, DEFAULT_DURATION_DECIMALS)
    meta = _format_analysis_metadata(analyzed, risk_counts, duration, cat_counts)
    steps = next_steps if next_steps else list(DEFAULT_POSSIBLE_NEXT_STEPS)
    return {
        "status": "success", "metadata": meta, "clauses": analyzed,
        "checklist": generate_lawyer_checklist(analyzed), "possible_next_steps": steps,
    }


def analyze_document(
    source: Any,
    filename_or_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Main pipeline to analyze an entire legal document.

    Args:
        source: Filepath, raw text string, or file object to analyze.
        filename_or_key: Optional filename label or API key.
        gemini_api_key: Optional Google Gemini API key.

    Returns:
        Dict[str, Any]: Dictionary containing status, metadata, clauses, checklist, possible_next_steps.

    Raises:
        None.
    """
    start = time.time()
    api_key, filename = _resolve_api_key_and_filename(filename_or_key, gemini_api_key)
    doc_text = source if isinstance(source, str) and ("\n" in source or len(source) > 260) else extract_document_text(source)
    try:
        gemini_res = call_gemini_api(build_analysis_prompt(doc_text, filename), api_key)
    except Exception as e:
        return {"status": "error", "error": f"Gemini API failure: {str(e)}", "metadata": {}, "clauses": [], "checklist": [], "possible_next_steps": []}
    analyzed, cat_counts, risk_counts = _assemble_clauses(extract_clauses(doc_text))
    return _format_analysis_result(analyzed, cat_counts, risk_counts, start, _extract_next_steps(gemini_res))


def _extract_risk_counts(clauses: List[Any]) -> Dict[str, int]:
    """Tallies risk levels across clause list.

    Args:
        clauses: List of clause dictionaries.

    Returns:
        Dict[str, int]: Tally of high, medium, and low clauses.

    Raises:
        None.
    """
    counts = {"high": 0, "medium": 0, "low": 0}
    for c in clauses:
        lvl = c.get("risk", {}).get("level", "low") if isinstance(c.get("risk"), dict) else c.get("risk", "low") if isinstance(c.get("risk"), str) else "low"
        counts[lvl if lvl in counts else "medium"] += 1
    return counts


def calculate_risk_summary(analysis_result: Any) -> Dict[str, Any]:
    """Calculates risk summary metrics from an analysis result or clauses list.

    Args:
        analysis_result: Analysis dictionary or clause list.

    Returns:
        Dict[str, Any]: Summary dictionary with counts, scores, and labels.

    Raises:
        None.
    """
    clauses = analysis_result if isinstance(analysis_result, list) else analysis_result.get("clauses", []) if isinstance(analysis_result, dict) else []
    counts = _extract_risk_counts(clauses)
    meta = analysis_result.get("metadata", {}) if isinstance(analysis_result, dict) else {}
    score = meta.get("overall_score")
    level = meta.get("overall_level")
    label = meta.get("overall_label", "")
    if score is None or level is None:
        score, level, label = _compute_overall_risk(len(clauses), counts)
    return {
        "total_clauses": len(clauses),
        "high_risk_count": counts["high"], "medium_risk_count": counts["medium"], "low_risk_count": counts["low"],
        "risk_counts": counts, "overall_score": score, "overall_level": level, "overall_label": label,
    }


def _read_txt_from_stream(source: Any) -> str:
    """Reads plain text from a file-like stream.

    Args:
        source: File-like object.

    Returns:
        str: Decoded string.

    Raises:
        None.
    """
    content = source.read()
    if hasattr(source, "seek"):
        source.seek(0)
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)


def extract_text_from_txt(source: Any) -> str:
    """Extract plain text from a .txt file path or file-like object.

    Args:
        source: Filepath string or file-like object.

    Returns:
        str: Extracted plain text string, truncated to MAX_DOCUMENT_CHARS.

    Raises:
        FileNotFoundError: If the source filepath does not exist.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            if "\n" in source or len(source) > 200:
                return source[:MAX_DOCUMENT_CHARS]
            raise FileNotFoundError(f"File not found: {source}")
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:MAX_DOCUMENT_CHARS]
    if hasattr(source, "read"):
        return _read_txt_from_stream(source)[:MAX_DOCUMENT_CHARS]
    return str(source)[:MAX_DOCUMENT_CHARS]


def _read_pdf(source: str) -> str:
    """Reads text from PDF path.

    Args:
        source: PDF filepath.

    Returns:
        str: Extracted text.

    Raises:
        RuntimeError: If PDF reading fails.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(source)
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text from {source}: {e}")


def _read_docx(source: str) -> str:
    """Reads text from DOCX path.

    Args:
        source: DOCX filepath.

    Returns:
        str: Extracted text.

    Raises:
        RuntimeError: If DOCX reading fails.
    """
    try:
        import docx
        doc = docx.Document(source)
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise RuntimeError(f"Failed to extract DOCX text from {source}: {e}")


def _extract_text_from_path(source: str) -> str:
    """Dispatches text extraction for file path.

    Args:
        source: Filepath string.

    Returns:
        str: Extracted document text.

    Raises:
        ValueError: If file extension is unsupported.
        FileNotFoundError: If file not found on disk.
    """
    ext_match = re.search(r"\.[a-zA-Z0-9]{2,6}$", source.strip())
    ext = ext_match.group(0).lower() if ext_match else os.path.splitext(source)[1].lower()
    if ext and ext not in ALLOWED_DOC_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf, .docx, and .txt are supported.")
    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")
    if ext == ".pdf":
        return _read_pdf(source)
    if ext == ".docx":
        return _read_docx(source)
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_document_text(source: Any) -> str:
    """Extract plain text from a filepath or file object.

    Args:
        source: A string filepath, string raw text, or file storage object.

    Returns:
        str: Extracted plain text string truncated to MAX_DOCUMENT_CHARS.

    Raises:
        ValueError: If file extension is unsupported.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If binary document reading fails.
    """
    if isinstance(source, str) and ("\n" in source or len(source) > 260 or (not os.path.exists(source) and not re.search(r"\.[a-zA-Z0-9]{2,6}$", source.strip()))):
        return source[:MAX_DOCUMENT_CHARS]
    if isinstance(source, str):
        return _extract_text_from_path(source)[:MAX_DOCUMENT_CHARS]
    ext = os.path.splitext(getattr(source, "filename", "") or "")[1].lower()
    if ext and ext not in ALLOWED_DOC_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf, .docx, and .txt are supported.")
    return extract_text_from_file(source)[:MAX_DOCUMENT_CHARS]


def _analysis_json_schema() -> str:
    """Returns the expected JSON schema string for Gemini prompt.

    Returns:
        str: JSON schema text.

    Raises:
        None.
    """
    return (
        '{\n'
        '  "document_type": "string", "overall_summary": "string",\n'
        '  "risk_level": "high | medium | low",\n'
        '  "clauses": [{\n'
        '    "title": "string", "category": "string", "clause_text": "string",\n'
        '    "risk_level": "high | medium | low", "plain_english": "string",\n'
        '    "risk_reasons": ["string"]\n'
        '  }],\n'
        '  "lawyer_questions": ["string"],\n'
        '  "action_checklist": ["string"],\n'
        '  "possible_next_steps": ["string"]\n'
        '}'
    )


def build_analysis_prompt(document_text: str, filename: Optional[str] = None) -> str:
    """Builds the Gemini analysis prompt with structured JSON schema.

    Args:
        document_text: Text content of document to analyze.
        filename: Optional filename label.

    Returns:
        str: Formatted prompt string requiring strict JSON output.

    Raises:
        None.
    """
    doc_label = f'"{filename}"' if filename else "the uploaded document"
    return (
        f"You are a legal document analysis assistant (NOT a lawyer).\n"
        f"Analyze {doc_label} clause-by-clause.\n\n"
        f"IMPORTANT DISCLAIMER & INSTRUCTIONS:\n{LEGAL_DISCLAIMER}\n"
        f"Generate 2-4 generic next-step paths a user could consider (e.g. 'Negotiate the clause', "
        f"'Accept as-is', 'Seek clarification before signing', 'Walk away from this term') "
        f"framed strictly as informational paths, not legal recommendations.\n\n"
        f"DOCUMENT TEXT:\n---\n{document_text[:MAX_DOCUMENT_CHARS]}\n---\n\n"
        f"Analyze the clauses and provide your output strictly as a JSON object matching this schema:\n"
        f"{_analysis_json_schema()}\n\n"
        f"Respond with ONLY the JSON object, no explanation text before or after."
    )
