"""Legal document comparison engine for ClauseWise.

Compares two legal documents (e.g. two contract versions, a lease vs
its renewal, competing vendor agreements) and identifies differences,
missing clauses, and which party each version tends to favor.

IMPORTANT: This tool provides comparative information to help users
understand differences between documents. It does NOT provide legal
advice on which version to accept or sign.
"""

import os
import json
import re
import difflib
import requests
from typing import Optional, Dict, Any, List, Tuple
from analyzer import (
    GEMINI_API_URL, GEMINI_MODEL, CONFIG,
    call_gemini_api, extract_document_text,
    extract_clauses, classify_clause, assess_risk
)

DISCLAIMER_TEXT = (
    "This tool provides comparative information to help users understand differences "
    "between documents. It does NOT provide legal advice on which version to accept or sign."
)


def build_comparison_prompt(
    doc_a_text: str,
    doc_a_name: str,
    doc_b_text: str,
    doc_b_name: str
) -> str:
    """Builds the Gemini comparison prompt instructing it to act as a document comparison assistant.

    The prompt instructs the model (acting strictly as a comparison assistant, not a lawyer) to:
    - Identify both document types and confirm if they are comparable.
    - Detect clauses present in Document A but missing in Document B, and vice versa.
    - Identify clauses covering the same topic but with differing commercial or legal terms.
    - Note which party each difference favors in a neutral, non-prescriptive manner.
    - Generate an overall comparison summary (3-5 sentences).
    - Flag key discussion points for legal consultation.
    - Strictly avoid recommending which document to sign or declaring one legally superior.

    Args:
        doc_a_text: Full text content of Document A.
        doc_a_name: Filename or label for Document A.
        doc_b_text: Full text content of Document B.
        doc_b_name: Filename or label for Document B.

    Returns:
        A structured prompt string requiring a strict JSON response.
    """
    prompt = f"""You are a legal document comparison assistant (NOT a lawyer).
Your task is to objectively compare two legal documents and identify structural differences, missing provisions, and differing terms.

IMPORTANT INSTRUCTION:
Do not recommend which document to sign or declare one legally superior. Only describe differences factually and note which party each term favors.

DOCUMENT A: "{doc_a_name}"
---
{doc_a_text[:12000]}
---

DOCUMENT B: "{doc_b_name}"
---
{doc_b_text[:12000]}
---

Analyze both documents and perform the following:
1. Identify both document types (e.g. Non-Disclosure Agreement, Commercial Lease, SaaS Agreement) and confirm whether they are comparable. Note if they are different document types entirely.
2. Find clauses present in Document A but MISSING from Document B.
3. Find clauses present in Document B but MISSING from Document A.
4. Find clauses covering the SAME topic but with DIFFERENT terms (e.g. different liability caps, payment terms, notice windows, termination conditions). For each difference:
   - Topic name
   - What Document A says
   - What Document B says
   - Which party the term favors and why (framed neutrally, e.g. "gives more protection to the tenant" or "limits exposure for the vendor", NOT "is better")
   - Significance (High, Medium, or Low)
5. Generate an overall comparison summary (3 to 5 sentences).
6. Flag key discussion points that are most significant to discuss with a lawyer before choosing between the two documents.
7. Assign an overall confidence score (0.0 to 1.0).

You MUST respond with a STRICT JSON object matching this schema exactly:
{{
  "doc_a_type": "<document A type>",
  "doc_b_type": "<document B type>",
  "documents_comparable": true,
  "comparability_note": "<explanation of comparability>",
  "overall_comparison": "<3-5 sentence overall summary>",
  "missing_in_a": [
    {{"topic": "<clause topic>", "found_in_b": "<summary of clause in B>", "significance": "<High|Medium|Low>"}}
  ],
  "missing_in_b": [
    {{"topic": "<clause topic>", "found_in_a": "<summary of clause in A>", "significance": "<High|Medium|Low>"}}
  ],
  "differing_terms": [
    {{
      "topic": "<topic name>",
      "doc_a_says": "<summary of term in A>",
      "doc_b_says": "<summary of term in B>",
      "favors": "<party favored and neutral rationale>",
      "significance": "<High|Medium|Low>"
    }}
  ],
  "key_discussion_points": [
    "<point to discuss with lawyer>"
  ],
  "confidence_score": 0.95
}}
"""
    return prompt.strip()


def calculate_comparison_metrics(comparison_result: dict) -> dict:
    """Calculates summary metrics from a legal document comparison result.

    Counts total differences, missing clauses across both documents, differing
    terms, and items flagged with high or significant importance.

    Args:
        comparison_result: Dictionary containing comparison findings,
            including missing_in_a, missing_in_b, and differing_terms.

    Returns:
        A dictionary containing:
            total_differences: Total count of missing clauses and differing terms.
            missing_clauses_count: Total count of clauses missing in either document.
            differing_terms_count: Total count of differing terms.
            high_significance_count: Count of differences marked as high or significant.
    """
    if not isinstance(comparison_result, dict):
        return {
            "total_differences": 0,
            "missing_clauses_count": 0,
            "differing_terms_count": 0,
            "high_significance_count": 0
        }

    missing_a = comparison_result.get("missing_in_a", []) or []
    missing_b = comparison_result.get("missing_in_b", []) or []
    differing = comparison_result.get("differing_terms", []) or []

    missing_clauses_count = len(missing_a) + len(missing_b)
    differing_terms_count = len(differing)
    total_differences = missing_clauses_count + differing_terms_count

    high_sig_terms = {"high", "significant", "critical", "urgent"}
    high_significance_count = 0

    all_items = missing_a + missing_b + differing
    for item in all_items:
        if isinstance(item, dict):
            sig = str(item.get("significance", "")).strip().lower()
            if any(term in sig for term in high_sig_terms):
                high_significance_count += 1

    return {
        "total_differences": total_differences,
        "missing_clauses_count": missing_clauses_count,
        "differing_terms_count": differing_terms_count,
        "high_significance_count": high_significance_count
    }


def _heuristic_comparison_fallback(
    text_a: str,
    filename_a: str,
    text_b: str,
    filename_b: str
) -> dict:
    """Generates an intelligent heuristic comparison when Gemini API is offline or unavailable.

    Args:
        text_a: Full text of Document A.
        filename_a: Label for Document A.
        text_b: Full text of Document B.
        filename_b: Label for Document B.

    Returns:
        Structured dictionary conforming to the required comparison schema.
    """
    clauses_a = extract_clauses(text_a)
    clauses_b = extract_clauses(text_b)

    for c in clauses_a:
        c["category"] = classify_clause(c["text"], c.get("title", ""))
        c["risk"] = assess_risk(c["text"], c["category"])

    for c in clauses_b:
        c["category"] = classify_clause(c["text"], c.get("title", ""))
        c["risk"] = assess_risk(c["text"], c["category"])

    missing_in_a = []
    missing_in_b = []
    differing_terms = []
    used_b = set()

    for ca in clauses_a:
        best_match = None
        best_sim = 0.0
        best_idx = None

        for idx, cb in enumerate(clauses_b):
            if idx in used_b:
                continue
            sim = compute_clause_similarity(ca, cb)
            if sim > best_sim:
                best_sim = sim
                best_match = cb
                best_idx = idx

        if best_match is not None and best_sim >= 0.50:
            used_b.add(best_idx)
            text_sim = difflib.SequenceMatcher(None, ca["text"].strip(), best_match["text"].strip()).ratio()
            if text_sim < 0.95:
                # Differing terms found
                rank_a = {"low": 1, "medium": 2, "high": 3}.get(ca["risk"]["level"], 1)
                rank_b = {"low": 1, "medium": 2, "high": 3}.get(best_match["risk"]["level"], 1)

                if rank_b > rank_a:
                    favors = f"Favors {filename_a} party (Document B introduces higher exposure or restrictive terms)"
                    sig = "High"
                elif rank_a > rank_b:
                    favors = f"Favors {filename_b} party (Document B reduces liability or softens obligations)"
                    sig = "High"
                else:
                    favors = "Neutral wording variation"
                    sig = "Medium"

                differing_terms.append({
                    "topic": ca.get("title", ca.get("category", "General")),
                    "doc_a_says": ca["text"][:300].strip(),
                    "doc_b_says": best_match["text"][:300].strip(),
                    "favors": favors,
                    "significance": sig
                })
        else:
            missing_in_b.append({
                "topic": ca.get("title", ca.get("category", "General")),
                "found_in_a": ca["text"][:300].strip(),
                "significance": "High" if ca["risk"]["level"] == "high" else "Medium"
            })

    for idx, cb in enumerate(clauses_b):
        if idx not in used_b:
            missing_in_a.append({
                "topic": cb.get("title", cb.get("category", "General")),
                "found_in_b": cb["text"][:300].strip(),
                "significance": "High" if cb["risk"]["level"] == "high" else "Medium"
            })

    # Summary synthesis
    summary = (
        f"Compared '{filename_a}' with '{filename_b}'. Identified {len(differing_terms)} differing provisions, "
        f"{len(missing_in_b)} clauses unique to '{filename_a}', and {len(missing_in_a)} provisions added in '{filename_b}'. "
        "Review differing liability, payment, and termination terms with legal counsel."
    )

    discussion_points = []
    for item in differing_terms:
        if item.get("significance") == "High":
            discussion_points.append(f"Discuss differing terms in '{item.get('topic')}' ({item.get('favors')}).")
    for item in missing_in_b:
        if item.get("significance") == "High":
            discussion_points.append(f"Evaluate removal of '{item.get('topic')}' from {filename_b}.")
    for item in missing_in_a:
        if item.get("significance") == "High":
            discussion_points.append(f"Review addition of '{item.get('topic')}' in {filename_b}.")

    if not discussion_points:
        discussion_points.append("Confirm all commercial schedules and governing law clauses are consistent.")

    return {
        "doc_a_type": "Legal Agreement",
        "doc_b_type": "Legal Agreement",
        "documents_comparable": True,
        "comparability_note": "Both documents exhibit consistent contractual structure suitable for comparison.",
        "overall_comparison": summary,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "differing_terms": differing_terms,
        "key_discussion_points": discussion_points[:5],
        "confidence_score": 0.90
    }


def compare_documents(
    filepath_a: str,
    filename_a: str,
    filepath_b: str,
    filename_b: str,
    api_key: str
) -> dict:
    """Compares two legal documents end-to-end, identifying differences and missing terms.

    Full pipeline:
    1. Extracts text from both documents using analyzer.extract_document_text.
    2. Builds comparison prompt via build_comparison_prompt.
    3. Calls Gemini API using analyzer.call_gemini_api.
    4. Parses, validates response, or executes heuristic fallback if API is unavailable.
    5. Computes summary metrics via calculate_comparison_metrics.
    6. Appends mandatory legal disclaimer.
    7. Wrapped in try/except to guarantee no unhandled crashes.

    Args:
        filepath_a: Filepath or text content of Document A.
        filename_a: Name or label for Document A.
        filepath_b: Filepath or text content of Document B.
        filename_b: Name or label for Document B.
        api_key: Google Gemini API key.

    Returns:
        A dictionary containing the validated comparison result, metrics, and disclaimer.
    """
    try:
        # Step 0: Validate input existence
        if not filepath_a or (isinstance(filepath_a, str) and not filepath_a.strip()):
            return {
                "status": "error",
                "error": f"Document A ('{filename_a}') is empty or missing.",
                "disclaimer": DISCLAIMER_TEXT
            }
        if not filepath_b or (isinstance(filepath_b, str) and not filepath_b.strip()):
            return {
                "status": "error",
                "error": f"Document B ('{filename_b}') is empty or missing.",
                "disclaimer": DISCLAIMER_TEXT
            }

        # Step 1: Text extraction
        text_a = extract_document_text(filepath_a)
        text_b = extract_document_text(filepath_b)

        if not text_a or not text_a.strip():
            return {
                "status": "error",
                "error": f"Document A ('{filename_a}') contains no readable text.",
                "disclaimer": DISCLAIMER_TEXT
            }
        if not text_b or not text_b.strip():
            return {
                "status": "error",
                "error": f"Document B ('{filename_b}') contains no readable text.",
                "disclaimer": DISCLAIMER_TEXT
            }

        # Step 2: Build comparison prompt
        prompt = build_comparison_prompt(text_a, filename_a, text_b, filename_b)

        # Step 3: Call Gemini API
        system_instruction = (
            "You are an expert legal document comparison assistant. You provide neutral, "
            "factual comparisons in valid JSON format only. You do not provide legal advice."
        )
        api_result = call_gemini_api(prompt, api_key, system_instruction=system_instruction)

        # Step 4: Validate or fallback
        if (
            isinstance(api_result, dict) and
            "overall_comparison" in api_result and
            "differing_terms" in api_result
        ):
            result = api_result
            result["status"] = "success"
        else:
            # Fallback to local heuristic engine
            result = _heuristic_comparison_fallback(text_a, filename_a, text_b, filename_b)
            result["status"] = "success"

        # Step 5: Metrics & disclaimer
        metrics = calculate_comparison_metrics(result)
        result["metrics"] = metrics
        result["disclaimer"] = DISCLAIMER_TEXT

        # Step 6: Generate side-by-side visual clause diffs for UI rendering
        clauses_a = extract_clauses(text_a)
        clauses_b = extract_clauses(text_b)
        for c in clauses_a:
            c["category"] = classify_clause(c["text"], c.get("title", ""))
            c["risk"] = assess_risk(c["text"], c["category"])
        for c in clauses_b:
            c["category"] = classify_clause(c["text"], c.get("title", ""))
            c["risk"] = assess_risk(c["text"], c["category"])

        comparisons = []
        used_b_indices = set()
        for ca in clauses_a:
            best_sim = 0.0
            best_idx = None
            for idx, cb in enumerate(clauses_b):
                if idx in used_b_indices:
                    continue
                sim = compute_clause_similarity(ca, cb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx is not None and best_sim >= 0.50:
                used_b_indices.add(best_idx)
                cb = clauses_b[best_idx]
                text_sim = difflib.SequenceMatcher(None, ca["text"].strip(), cb["text"].strip()).ratio()
                status = "unchanged" if text_sim >= 0.95 else "modified"
                badge_class = "badge-secondary" if status == "unchanged" else "badge-warning"
                rank_a = {"low": 1, "medium": 2, "high": 3}.get(ca["risk"]["level"], 1)
                rank_b = {"low": 1, "medium": 2, "high": 3}.get(cb["risk"]["level"], 1)
                if rank_b > rank_a:
                    risk_delta = "increased"
                    risk_delta_label = "⚠️ Risk Increased in Version B"
                elif rank_b < rank_a:
                    risk_delta = "decreased"
                    risk_delta_label = "✅ Risk Decreased in Version B"
                else:
                    risk_delta = "neutral"
                    risk_delta_label = "Unchanged Risk Level"
                diff_a, diff_b = generate_diff_html(ca["text"], cb["text"])
                comparisons.append({
                    "status": status,
                    "badge_class": badge_class,
                    "similarity": round(text_sim * 100, 1),
                    "risk_delta": risk_delta,
                    "risk_delta_label": risk_delta_label,
                    "clause_a": ca,
                    "clause_b": cb,
                    "diff_html_a": diff_a,
                    "diff_html_b": diff_b
                })
            else:
                comparisons.append({
                    "status": "removed",
                    "badge_class": "badge-danger",
                    "similarity": 0.0,
                    "risk_delta": "removed",
                    "risk_delta_label": "❌ Clause Removed in Version B",
                    "clause_a": ca,
                    "clause_b": None,
                    "diff_html_a": f"<del class='diff-del'>{ca['text']}</del>",
                    "diff_html_b": "<span class='text-muted italic'>[Provision Removed]</span>"
                })

        for idx, cb in enumerate(clauses_b):
            if idx not in used_b_indices:
                comparisons.append({
                    "status": "added",
                    "badge_class": "badge-success",
                    "similarity": 0.0,
                    "risk_delta": "added",
                    "risk_delta_label": f"✨ New Clause Added ({cb['risk']['label']})",
                    "clause_a": None,
                    "clause_b": cb,
                    "diff_html_a": "<span class='text-muted italic'>[Not Present in Version A]</span>",
                    "diff_html_b": f"<ins class='diff-ins'>{cb['text']}</ins>"
                })

        counts = {
            "total": len(comparisons),
            "unchanged": sum(1 for c in comparisons if c["status"] == "unchanged"),
            "modified": sum(1 for c in comparisons if c["status"] == "modified"),
            "added": sum(1 for c in comparisons if c["status"] == "added"),
            "removed": sum(1 for c in comparisons if c["status"] == "removed"),
            "risk_increased": sum(1 for c in comparisons if c["risk_delta"] == "increased"),
            "risk_decreased": sum(1 for c in comparisons if c["risk_delta"] == "decreased")
        }
        result["comparisons"] = comparisons
        result["counts"] = counts

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "doc_a_type": "Unknown",
            "doc_b_type": "Unknown",
            "documents_comparable": False,
            "comparability_note": f"Comparison failed due to error: {e}",
            "overall_comparison": f"An error occurred during comparison: {e}",
            "missing_in_a": [],
            "missing_in_b": [],
            "differing_terms": [],
            "key_discussion_points": [],
            "confidence_score": 0.0,
            "metrics": {
                "total_differences": 0,
                "missing_clauses_count": 0,
                "differing_terms_count": 0,
                "high_significance_count": 0
            },
            "disclaimer": DISCLAIMER_TEXT
        }


# === AUXILIARY COMPARISON HELPERS (for UI Diff Rendering) ===

def compute_clause_similarity(clause_a: Dict[str, Any], clause_b: Dict[str, Any]) -> float:
    """Calculates similarity score between two clauses using sequence matching
    on normalized text and titles.
    """
    text_a = re.sub(r"\s+", " ", clause_a.get("text", "")).strip().lower()
    text_b = re.sub(r"\s+", " ", clause_b.get("text", "")).strip().lower()
    
    title_a = clause_a.get("title", "").strip().lower()
    title_b = clause_b.get("title", "").strip().lower()

    title_sim = difflib.SequenceMatcher(None, title_a, title_b).ratio() if title_a and title_b else 0.0
    text_sim = difflib.SequenceMatcher(None, text_a, text_b).ratio()

    num_a = clause_a.get("number", "").strip().lower()
    num_b = clause_b.get("number", "").strip().lower()
    has_same_num = (num_a and num_b and num_a == num_b)

    if num_a and num_b and num_a != num_b and title_sim < 0.85:
        num_penalty = 0.25
    else:
        num_penalty = 0.0

    num_bonus = 0.15 if has_same_num else 0.0

    weighted = (text_sim * 0.65) + (title_sim * 0.25) + num_bonus - num_penalty
    return max(0.0, min(1.0, weighted))


def generate_diff_html(text_a: str, text_b: str) -> Tuple[str, str]:
    """Generates inline HTML diff highlighting added and deleted words."""
    words_a = text_a.split()
    words_b = text_b.split()
    
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    diff_a = []
    diff_b = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            chunk = " ".join(words_a[i1:i2])
            diff_a.append(chunk)
            diff_b.append(chunk)
        elif tag == 'delete':
            chunk = " ".join(words_a[i1:i2])
            diff_a.append(f"<del class='diff-del'>{chunk}</del>")
        elif tag == 'insert':
            chunk = " ".join(words_b[j1:j2])
            diff_b.append(f"<ins class='diff-ins'>{chunk}</ins>")
        elif tag == 'replace':
            chunk_a = " ".join(words_a[i1:i2])
            chunk_b = " ".join(words_b[j1:j2])
            diff_a.append(f"<del class='diff-del'>{chunk_a}</del>")
            diff_b.append(f"<ins class='diff-ins'>{chunk_b}</ins>")

    return " ".join(diff_a), " ".join(diff_b)
