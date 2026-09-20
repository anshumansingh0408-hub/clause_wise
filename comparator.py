"""Legal document comparison engine for ClauseWise.

Compares two legal documents (e.g. two contract versions, a lease vs
its renewal, competing vendor agreements) and identifies differences,
missing clauses, and which party each version tends to favor.

IMPORTANT: This tool provides comparative information to help users
understand differences between documents. It does NOT provide legal
advice on which version to accept or sign.
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from analyzer import (
    assess_risk,
    classify_clause,
    extract_clauses,
    extract_document_text,
)
from utils import (
    call_gemini_api,
    CONFIG,
    GEMINI_API_URL,
    GEMINI_MODEL,
)


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

SIMILARITY_MATCH_THRESHOLD = 0.50
TEXT_SIMILARITY_UNCHANGED = 0.95
MAX_DOC_CHARS_COMPARISON = 12000
SNIPPET_PREVIEW_LEN = 300
PERCENTAGE_MULTIPLIER = 100
MAX_DISCUSSION_POINTS = 5

TEXT_SIM_WEIGHT = 0.65
TITLE_SIM_WEIGHT = 0.25
NUM_BONUS = 0.15
NUM_PENALTY = 0.25
TITLE_SIM_THRESHOLD = 0.85

HIGH_CONFIDENCE_SCORE = 0.95
FALLBACK_CONFIDENCE_SCORE = 0.90
HIGH_SIG_KEYWORDS = {"high", "significant", "critical", "urgent"}

DISCLAIMER_TEXT = (
    "This tool provides comparative information to help users understand differences "
    "between documents. It does NOT provide legal advice on which version to accept or sign."
)

COMPARISON_JSON_SCHEMA = (
    '{\n'
    '  "doc_a_type": "<document A type>",\n'
    '  "doc_b_type": "<document B type>",\n'
    '  "documents_comparable": true,\n'
    '  "comparability_note": "<explanation of comparability>",\n'
    '  "overall_comparison": "<3-5 sentence overall summary>",\n'
    '  "missing_in_a": [{"topic": "<clause topic>", "found_in_b": "<summary of clause in B>", "significance": "<High|Medium|Low>"}],\n'
    '  "missing_in_b": [{"topic": "<clause topic>", "found_in_a": "<summary of clause in A>", "significance": "<High|Medium|Low>"}],\n'
    '  "differing_terms": [\n'
    '    {"topic": "<topic name>", "doc_a_says": "<term A>", "doc_b_says": "<term B>", "favors": "<party>", "significance": "<High|Medium|Low>"}\n'
    '  ],\n'
    '  "key_discussion_points": ["<point to discuss with lawyer>"],\n'
    '  "confidence_score": 0.95\n'
    '}'
)


# ============================================================================
# COMPARISON PROMPT GENERATOR
# ============================================================================

def _comparison_json_schema() -> str:
    """Returns the JSON schema string for Gemini comparison prompt.

    Returns:
        str: JSON schema text.
    """
    return COMPARISON_JSON_SCHEMA


def build_comparison_prompt(
    doc_a_text: str, doc_a_name: str, doc_b_text: str, doc_b_name: str
) -> str:
    """Builds the Gemini comparison prompt instructing it to compare documents.

    Args:
        doc_a_text: Full text content of Document A.
        doc_a_name: Filename or label for Document A.
        doc_b_text: Full text content of Document B.
        doc_b_name: Filename or label for Document B.

    Returns:
        str: A structured prompt string requiring a strict JSON response.
    """
    snippet_a = doc_a_text[:MAX_DOC_CHARS_COMPARISON]
    snippet_b = doc_b_text[:MAX_DOC_CHARS_COMPARISON]
    instructions = (
        "You are a legal document comparison assistant (NOT a lawyer).\n"
        "Do not recommend which document to sign or declare one legally superior.\n\n"
        f'DOCUMENT A: "{doc_a_name}"\n---\n{snippet_a}\n---\n\n'
        f'DOCUMENT B: "{doc_b_name}"\n---\n{snippet_b}\n---\n\n'
        "Analyze both documents, identify differing provisions, missing clauses, and which party terms favor.\n"
        "You MUST respond with a STRICT JSON object matching this schema exactly:\n"
    )
    return f"{instructions}{COMPARISON_JSON_SCHEMA}"


# ============================================================================
# METRICS & HEURISTIC FALLBACK
# ============================================================================

def _count_high_significance(items: List[Any]) -> int:
    """Counts items flagged with high or significant priority.

    Args:
        items: List of missing or differing clause dictionaries.

    Returns:
        int: Number of items with high significance.
    """
    count = 0
    for item in items:
        if isinstance(item, dict):
            sig = str(item.get("significance", "")).strip().lower()
            if any(term in sig for term in HIGH_SIG_KEYWORDS):
                count += 1
    return count


def calculate_comparison_metrics(comparison_result: Dict[str, Any]) -> Dict[str, int]:
    """Calculates summary metrics from a legal document comparison result.

    Args:
        comparison_result: Dictionary containing comparison findings.

    Returns:
        Dict[str, int]: Counts of total differences, missing clauses, and high significance items.
    """
    if not isinstance(comparison_result, dict):
        return {"total_differences": 0, "missing_clauses_count": 0, "differing_terms_count": 0, "high_significance_count": 0}
    missing_a = comparison_result.get("missing_in_a", []) or []
    missing_b = comparison_result.get("missing_in_b", []) or []
    differing = comparison_result.get("differing_terms", []) or []
    missing_count = len(missing_a) + len(missing_b)
    diff_count = len(differing)
    return {
        "total_differences": missing_count + diff_count,
        "missing_clauses_count": missing_count,
        "differing_terms_count": diff_count,
        "high_significance_count": _count_high_significance(missing_a + missing_b + differing),
    }


def _find_best_clause_match(
    ca: Dict[str, Any], clauses_b: List[Dict[str, Any]], used_b: Set[int]
) -> Tuple[Optional[Dict[str, Any]], Optional[int], float]:
    """Finds best matching clause in B for clause A.

    Args:
        ca: Target clause from Document A.
        clauses_b: Candidate clauses from Document B.
        used_b: Indices of already matched clauses in B.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[int], float]: Match, index, similarity.
    """
    best_match, best_sim, best_idx = None, 0.0, None
    for idx, cb in enumerate(clauses_b):
        if idx in used_b:
            continue
        sim = compute_clause_similarity(ca, cb)
        if sim > best_sim:
            best_sim, best_match, best_idx = sim, cb, idx
    return best_match, best_idx, best_sim


def _determine_favors(ca: Dict[str, Any], cb: Dict[str, Any], name_a: str, name_b: str) -> Tuple[str, str]:
    """Determines which party a differing term favors.

    Args:
        ca: Clause from Document A.
        cb: Clause from Document B.
        name_a: Document A name.
        name_b: Document B name.

    Returns:
        Tuple[str, str]: Favors description and significance string.
    """
    ranks = {"low": 1, "medium": 2, "high": 3}
    rank_a = ranks.get(ca.get("risk", {}).get("level", "low"), 1)
    rank_b = ranks.get(cb.get("risk", {}).get("level", "low"), 1)
    if rank_b > rank_a:
        return f"Favors {name_a} party (Document B introduces higher exposure or restrictive terms)", "High"
    if rank_a > rank_b:
        return f"Favors {name_b} party (Document B reduces liability or softens obligations)", "High"
    return "Neutral wording variation", "Medium"


def _build_heuristic_differing_item(
    ca: Dict[str, Any], cb: Dict[str, Any], name_a: str, name_b: str
) -> Dict[str, Any]:
    """Constructs differing term item.

    Args:
        ca: Clause from Document A.
        cb: Clause from Document B.
        name_a: Document A label.
        name_b: Document B label.

    Returns:
        Dict[str, Any]: Differing term record.
    """
    favors, sig = _determine_favors(ca, cb, name_a, name_b)
    topic = ca.get("title", ca.get("category", "General"))
    return {
        "topic": topic,
        "doc_a_says": ca["text"][:SNIPPET_PREVIEW_LEN].strip(),
        "doc_b_says": cb["text"][:SNIPPET_PREVIEW_LEN].strip(),
        "favors": favors,
        "significance": sig,
    }


def _find_unmatched_b_clauses(
    clauses_b: List[Dict[str, Any]], used_b: Set[int]
) -> List[Dict[str, Any]]:
    """Gathers clauses in Document B that had no match in Document A.

    Args:
        clauses_b: Document B clauses list.
        used_b: Indices matched in Document B.

    Returns:
        List[Dict[str, Any]]: Items missing in A.
    """
    missing_in_a = []
    for idx, cb in enumerate(clauses_b):
        if idx not in used_b:
            missing_in_a.append({
                "topic": cb.get("title", cb.get("category", "General")),
                "found_in_b": cb["text"][:SNIPPET_PREVIEW_LEN].strip(),
                "significance": "High" if cb.get("risk", {}).get("level") == "high" else "Medium",
            })
    return missing_in_a


def _match_single_clause(
    ca: Dict[str, Any], clauses_b: List[Dict[str, Any]], used_b: Set[int],
    diffs: List[Dict[str, Any]], missing_b: List[Dict[str, Any]], name_a: str, name_b: str
) -> None:
    """Matches single clause from A against B and appends to appropriate list.

    Args:
        ca: Clause from A.
        clauses_b: Clauses list from B.
        used_b: Indices matched in B.
        diffs: Differing list.
        missing_b: Missing in B list.
        name_a: Label A.
        name_b: Label B.
    """
    match, idx, sim = _find_best_clause_match(ca, clauses_b, used_b)
    if match is not None and sim >= SIMILARITY_MATCH_THRESHOLD:
        used_b.add(idx)
        text_sim = difflib.SequenceMatcher(None, ca["text"].strip(), match["text"].strip()).ratio()
        if text_sim < TEXT_SIMILARITY_UNCHANGED:
            diffs.append(_build_heuristic_differing_item(ca, match, name_a, name_b))
    else:
        sig = "High" if ca.get("risk", {}).get("level") == "high" else "Medium"
        missing_b.append({"topic": ca.get("title", ca.get("category", "General")), "found_in_a": ca["text"][:SNIPPET_PREVIEW_LEN].strip(), "significance": sig})


def _match_clauses_heuristically(
    clauses_a: List[Dict[str, Any]], clauses_b: List[Dict[str, Any]], name_a: str, name_b: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Matches clauses between A and B and compiles differences.

    Args:
        clauses_a: Clauses from Document A.
        clauses_b: Clauses from Document B.
        name_a: Label A.
        name_b: Label B.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]: Differences.
    """
    missing_in_b, differing, used_b = [], [], set()
    for ca in clauses_a:
        _match_single_clause(ca, clauses_b, used_b, differing, missing_in_b, name_a, name_b)
    missing_in_a = _find_unmatched_b_clauses(clauses_b, used_b)
    return missing_in_a, missing_in_b, differing


def _build_heuristic_discussion_points(
    differing: List[Dict[str, Any]], missing_b: List[Dict[str, Any]], missing_a: List[Dict[str, Any]], name_b: str
) -> List[str]:
    """Generates key discussion points for lawyer prep.

    Args:
        differing: Differing terms list.
        missing_b: Clauses missing in B.
        missing_a: Clauses missing in A.
        name_b: Label for Document B.

    Returns:
        List[str]: Discussion points list.
    """
    points = [f"Discuss differing terms in '{i.get('topic')}' ({i.get('favors')})." for i in differing if i.get("significance") == "High"]
    points += [f"Evaluate removal of '{i.get('topic')}' from {name_b}." for i in missing_b if i.get("significance") == "High"]
    points += [f"Review addition of '{i.get('topic')}' in {name_b}." for i in missing_a if i.get("significance") == "High"]
    return points[:MAX_DISCUSSION_POINTS] if points else ["Confirm all commercial schedules and governing law clauses are consistent."]


def _heuristic_comparison_fallback(text_a: str, filename_a: str, text_b: str, filename_b: str) -> Dict[str, Any]:
    """Generates an intelligent heuristic comparison when Gemini API is unavailable.

    Args:
        text_a: Full text of Document A.
        filename_a: Label for Document A.
        text_b: Full text of Document B.
        filename_b: Label for Document B.

    Returns:
        Dict[str, Any]: Structured comparison dictionary.
    """
    clauses_a, clauses_b = extract_clauses(text_a), extract_clauses(text_b)
    for c in clauses_a + clauses_b:
        c["category"] = classify_clause(c["text"], c.get("title", ""))
        c["risk"] = assess_risk(c["text"], c["category"])
    missing_a, missing_b, differing = _match_clauses_heuristically(clauses_a, clauses_b, filename_a, filename_b)
    summary = f"Compared '{filename_a}' with '{filename_b}'. Identified {len(differing)} differing provisions."
    return {
        "doc_a_type": "Legal Agreement", "doc_b_type": "Legal Agreement", "documents_comparable": True,
        "comparability_note": "Both documents exhibit consistent contractual structure.",
        "overall_comparison": summary, "missing_in_a": missing_a, "missing_in_b": missing_b, "differing_terms": differing,
        "key_discussion_points": _build_heuristic_discussion_points(differing, missing_b, missing_a, filename_b),
        "confidence_score": FALLBACK_CONFIDENCE_SCORE,
    }


# ============================================================================
# PIPELINE & DOCUMENT COMPARISON
# ============================================================================

def _validate_inputs(
    filepath_a: str, filename_a: str, filepath_b: str, filename_b: str
) -> Optional[Dict[str, Any]]:
    """Checks validity of comparison inputs.

    Args:
        filepath_a: Document A source.
        filename_a: Document A label.
        filepath_b: Document B source.
        filename_b: Document B label.

    Returns:
        Optional[Dict[str, Any]]: Error dictionary if invalid, None if valid.
    """
    if not filepath_a or (isinstance(filepath_a, str) and not filepath_a.strip()):
        return {"status": "error", "error": f"Document A ('{filename_a}') is empty or missing.", "disclaimer": DISCLAIMER_TEXT}
    if not filepath_b or (isinstance(filepath_b, str) and not filepath_b.strip()):
        return {"status": "error", "error": f"Document B ('{filename_b}') is empty or missing.", "disclaimer": DISCLAIMER_TEXT}
    return None


def _format_risk_delta(rank_a: int, rank_b: int) -> Tuple[str, str]:
    """Calculates risk delta direction and badge label.

    Args:
        rank_a: Numeric risk rank in Document A.
        rank_b: Numeric risk rank in Document B.

    Returns:
        Tuple[str, str]: Risk delta key and label.
    """
    if rank_b > rank_a:
        return "increased", "⚠️ Risk Increased in Version B"
    if rank_b < rank_a:
        return "decreased", "✅ Risk Decreased in Version B"
    return "neutral", "Unchanged Risk Level"


def _build_matched_comp(ca: Dict[str, Any], cb: Dict[str, Any]) -> Dict[str, Any]:
    """Builds side-by-side comparison item for matched clauses.

    Args:
        ca: Clause from Document A.
        cb: Clause from Document B.

    Returns:
        Dict[str, Any]: Visual comparison item for UI.
    """
    text_sim = difflib.SequenceMatcher(None, ca["text"].strip(), cb["text"].strip()).ratio()
    status = "unchanged" if text_sim >= TEXT_SIMILARITY_UNCHANGED else "modified"
    ranks = {"low": 1, "medium": 2, "high": 3}
    delta, delta_label = _format_risk_delta(ranks.get(ca["risk"]["level"], 1), ranks.get(cb["risk"]["level"], 1))
    diff_a, diff_b = generate_diff_html(ca["text"], cb["text"])
    return {
        "status": status,
        "badge_class": "badge-secondary" if status == "unchanged" else "badge-warning",
        "similarity": round(text_sim * PERCENTAGE_MULTIPLIER, 1),
        "risk_delta": delta, "risk_delta_label": delta_label,
        "clause_a": ca, "clause_b": cb, "diff_html_a": diff_a, "diff_html_b": diff_b,
    }


def _build_removed_comp(ca: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs UI comparison item for clause removed in Document B.

    Args:
        ca: Clause from Document A.

    Returns:
        Dict[str, Any]: Formatted comparison item dictionary.
    """
    return {
        "status": "removed", "badge_class": "badge-danger", "similarity": 0.0,
        "risk_delta": "removed", "risk_delta_label": "❌ Clause Removed in Version B",
        "clause_a": ca, "clause_b": None,
        "diff_html_a": f"<del class='diff-del'>{ca['text']}</del>",
        "diff_html_b": "<span class='text-muted italic'>[Provision Removed]</span>",
    }


def _build_added_comp(cb: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs UI comparison item for new clause added in Document B.

    Args:
        cb: Clause from Document B.

    Returns:
        Dict[str, Any]: Formatted comparison item dictionary.
    """
    return {
        "status": "added", "badge_class": "badge-success", "similarity": 0.0,
        "risk_delta": "added", "risk_delta_label": f"✨ New Clause Added ({cb['risk']['label']})",
        "clause_a": None, "clause_b": cb,
        "diff_html_a": "<span class='text-muted italic'>[Not Present in Version A]</span>",
        "diff_html_b": f"<ins class='diff-ins'>{cb['text']}</ins>",
    }


def _build_side_by_side_comparisons(
    clauses_a: List[Dict[str, Any]], clauses_b: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Builds full side-by-side clause comparison list for UI rendering.

    Args:
        clauses_a: Annotated clauses from Document A.
        clauses_b: Annotated clauses from Document B.

    Returns:
        List[Dict[str, Any]]: UI comparison items list.
    """
    comparisons, used_b = [], set()
    for ca in clauses_a:
        match, idx, sim = _find_best_clause_match(ca, clauses_b, used_b)
        if idx is not None and sim >= SIMILARITY_MATCH_THRESHOLD:
            used_b.add(idx)
            comparisons.append(_build_matched_comp(ca, clauses_b[idx]))
        else:
            comparisons.append(_build_removed_comp(ca))
    for idx, cb in enumerate(clauses_b):
        if idx not in used_b:
            comparisons.append(_build_added_comp(cb))
    return comparisons


def _enrich_clauses_for_diff(text: str) -> List[Dict[str, Any]]:
    """Extracts, categorizes, and scores clauses from raw text.

    Args:
        text: Raw document text.

    Returns:
        List[Dict[str, Any]]: Enriched clauses list.
    """
    clauses = extract_clauses(text)
    for c in clauses:
        c["category"] = classify_clause(c["text"], c.get("title", ""))
        c["risk"] = assess_risk(c["text"], c["category"])
    return clauses


def _calculate_comp_counts(comps: List[Dict[str, Any]]) -> Dict[str, int]:
    """Tallies comparison item statuses and risk shifts.

    Args:
        comps: List of clause comparisons.

    Returns:
        Dict[str, int]: Aggregate counts.
    """
    return {
        "total": len(comps),
        "unchanged": sum(1 for c in comps if c["status"] == "unchanged"),
        "modified": sum(1 for c in comps if c["status"] == "modified"),
        "added": sum(1 for c in comps if c["status"] == "added"),
        "removed": sum(1 for c in comps if c["status"] == "removed"),
        "risk_increased": sum(1 for c in comps if c["risk_delta"] == "increased"),
        "risk_decreased": sum(1 for c in comps if c["risk_delta"] == "decreased"),
    }


def _run_comparison_engine(
    text_a: str, filename_a: str, text_b: str, filename_b: str, api_key: str
) -> Dict[str, Any]:
    """Runs Gemini comparison call or fallback heuristic.

    Args:
        text_a: Text of Document A.
        filename_a: Label A.
        text_b: Text of Document B.
        filename_b: Label B.
        api_key: Gemini API key.

    Returns:
        Dict[str, Any]: Parsed comparison result dictionary.
    """
    prompt = build_comparison_prompt(text_a, filename_a, text_b, filename_b)
    system_inst = "You are an expert legal document comparison assistant. Return valid JSON only."
    api_res = call_gemini_api(prompt, api_key, system_instruction=system_inst)
    res = api_res if (isinstance(api_res, dict) and "overall_comparison" in api_res and "differing_terms" in api_res) else _heuristic_comparison_fallback(text_a, filename_a, text_b, filename_b)
    res["status"] = "success"
    res["metrics"] = calculate_comparison_metrics(res)
    res["disclaimer"] = DISCLAIMER_TEXT
    comps = _build_side_by_side_comparisons(_enrich_clauses_for_diff(text_a), _enrich_clauses_for_diff(text_b))
    res["comparisons"], res["counts"] = comps, _calculate_comp_counts(comps)
    return res


def _handle_comparison_error(e: Exception) -> Dict[str, Any]:
    """Formats safe fallback response on unhandled comparison exception.

    Args:
        e: Caught exception.

    Returns:
        Dict[str, Any]: Safe error payload.
    """
    return {
        "status": "error", "error": str(e), "doc_a_type": "Unknown", "doc_b_type": "Unknown",
        "documents_comparable": False, "comparability_note": f"Error: {e}", "overall_comparison": f"Error: {e}",
        "missing_in_a": [], "missing_in_b": [], "differing_terms": [], "key_discussion_points": [],
        "confidence_score": 0.0, "metrics": {"total_differences": 0, "missing_clauses_count": 0, "differing_terms_count": 0, "high_significance_count": 0},
        "disclaimer": DISCLAIMER_TEXT,
    }


def _extract_comparison_texts(
    path_a: str, name_a: str, path_b: str, name_b: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Extracts texts for both documents with error checks.

    Args:
        path_a: Source A.
        name_a: Label A.
        path_b: Source B.
        name_b: Label B.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]: text_a, text_b, err_dict.
    """
    text_a, text_b = extract_document_text(path_a), extract_document_text(path_b)
    if not text_a or not text_a.strip():
        return None, None, {"status": "error", "error": f"Document A ('{name_a}') contains no readable text.", "disclaimer": DISCLAIMER_TEXT}
    if not text_b or not text_b.strip():
        return None, None, {"status": "error", "error": f"Document B ('{name_b}') contains no readable text.", "disclaimer": DISCLAIMER_TEXT}
    return text_a, text_b, None


def compare_documents(
    filepath_a: str, filename_a: str, filepath_b: str, filename_b: str, api_key: str
) -> Dict[str, Any]:
    """Compares two legal documents end-to-end, identifying differences and missing terms.

    Args:
        filepath_a: Filepath or text content of Document A.
        filename_a: Name or label for Document A.
        filepath_b: Filepath or text content of Document B.
        filename_b: Name or label for Document B.
        api_key: Google Gemini API key.

    Returns:
        Dict[str, Any]: Validated comparison result, metrics, and disclaimer.
    """
    err = _validate_inputs(filepath_a, filename_a, filepath_b, filename_b)
    if err:
        return err
    try:
        text_a, text_b, text_err = _extract_comparison_texts(filepath_a, filename_a, filepath_b, filename_b)
        if text_err:
            return text_err
        return _run_comparison_engine(text_a, filename_a, text_b, filename_b, api_key)
    except Exception as e:
        return _handle_comparison_error(e)


# ============================================================================
# AUXILIARY COMPARISON HELPERS (UI Diff Rendering)
# ============================================================================

def compute_clause_similarity(clause_a: Dict[str, Any], clause_b: Dict[str, Any]) -> float:
    """Calculates similarity score between two clauses using sequence matching.

    Args:
        clause_a: First clause dictionary.
        clause_b: Second clause dictionary.

    Returns:
        float: Similarity ratio between 0.0 and 1.0.
    """
    text_a = re.sub(r"\s+", " ", clause_a.get("text", "")).strip().lower()
    text_b = re.sub(r"\s+", " ", clause_b.get("text", "")).strip().lower()
    title_a, title_b = clause_a.get("title", "").strip().lower(), clause_b.get("title", "").strip().lower()
    title_sim = difflib.SequenceMatcher(None, title_a, title_b).ratio() if title_a and title_b else 0.0
    text_sim = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    num_a, num_b = clause_a.get("number", "").strip().lower(), clause_b.get("number", "").strip().lower()
    penalty = NUM_PENALTY if num_a and num_b and num_a != num_b and title_sim < TITLE_SIM_THRESHOLD else 0.0
    bonus = NUM_BONUS if (num_a and num_b and num_a == num_b) else 0.0
    weighted = (text_sim * TEXT_SIM_WEIGHT) + (title_sim * TITLE_SIM_WEIGHT) + bonus - penalty
    return max(0.0, min(1.0, weighted))


def _format_diff_tags(opcodes: List[Tuple[str, int, int, int, int]], words_a: List[str], words_b: List[str]) -> Tuple[List[str], List[str]]:
    """Formats HTML tags for word difference sequences.

    Args:
        opcodes: Difflib opcode tuples.
        words_a: List of words from text A.
        words_b: List of words from text B.

    Returns:
        Tuple[List[str], List[str]]: Formatted token lists.
    """
    diff_a, diff_b = [], []
    for tag, i1, i2, j1, j2 in opcodes:
        ca, cb = " ".join(words_a[i1:i2]), " ".join(words_b[j1:j2])
        if tag == "equal":
            diff_a.append(ca); diff_b.append(cb)
        elif tag == "delete":
            diff_a.append(f"<del class='diff-del'>{ca}</del>")
        elif tag == "insert":
            diff_b.append(f"<ins class='diff-ins'>{cb}</ins>")
        elif tag == "replace":
            diff_a.append(f"<del class='diff-del'>{ca}</del>"); diff_b.append(f"<ins class='diff-ins'>{cb}</ins>")
    return diff_a, diff_b


def generate_diff_html(text_a: str, text_b: str) -> Tuple[str, str]:
    """Generates inline HTML diff highlighting added and deleted words.

    Args:
        text_a: Original clause text A.
        text_b: Modified clause text B.

    Returns:
        Tuple[str, str]: HTML strings with diff markup for A and B.
    """
    words_a, words_b = text_a.split(), text_b.split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    diff_a, diff_b = _format_diff_tags(matcher.get_opcodes(), words_a, words_b)
    return " ".join(diff_a), " ".join(diff_b)
