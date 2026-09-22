"""Unit tests for ClauseWise legal document analyzer module."""

import unittest
from unittest.mock import patch
import tempfile
import os

from analyzer import (
    extract_text_from_txt,
    extract_document_text,
    build_analysis_prompt,
    analyze_document,
    calculate_risk_summary,
    call_gemini_api,
    extract_clauses,
    classify_clause,
    assess_risk,
    generate_plain_english_summary,
    generate_lawyer_checklist,
    CLAUSE_CATEGORIES,
    RISK_LEVELS,
    MAX_DOCUMENT_CHARS
)
from utils import SAMPLE_CONTRACTS


class TestAnalyzer(unittest.TestCase):
    """Test suite for legal document analyzer functionality."""

    def test_extract_text_from_txt(self):
        """Verify plain text extraction works from a text file."""
        content = "1. Term and Termination\nThis agreement lasts for 1 year."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name

        try:
            extracted = extract_text_from_txt(temp_path)
            self.assertEqual(extracted, content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_extract_document_text_unsupported_type(self):
        """Verify .xyz extension raises ValueError."""
        with self.assertRaises(ValueError):
            extract_document_text("unsupported_contract.xyz")

    def test_extract_document_text_truncates_long(self):
        """Verify text longer than MAX_DOCUMENT_CHARS gets truncated."""
        long_text = "A" * (MAX_DOCUMENT_CHARS + 500)
        extracted = extract_document_text(long_text)
        self.assertEqual(len(extracted), MAX_DOCUMENT_CHARS)

    def test_build_analysis_prompt_contains_schema(self):
        """Verify generated prompt contains key schema field names (e.g. 'risk_level', 'lawyer_questions', 'possible_next_steps')."""
        prompt = build_analysis_prompt("Sample agreement text", "contract.pdf")
        self.assertIn("risk_level", prompt)
        self.assertIn("lawyer_questions", prompt)
        self.assertIn("document_type", prompt)
        self.assertIn("overall_summary", prompt)
        self.assertIn("action_checklist", prompt)
        self.assertIn("possible_next_steps", prompt)

    def test_build_analysis_prompt_includes_disclaimer_instruction(self):
        """Verify prompt explicitly instructs against giving legal advice."""
        prompt = build_analysis_prompt("Sample agreement text")
        self.assertIn("not provide legal advice", prompt.lower())
        self.assertIn("not a lawyer", prompt.lower())

    @patch("analyzer.call_gemini_api", side_effect=Exception("API Connection Failure"))
    def test_analyze_document_handles_api_failure(self, mock_gemini):
        """Mock call_gemini_api to raise exception, verify analyze_document returns dict with 'error' key."""
        result = analyze_document("1. Term: 1 year.")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_calculate_risk_summary_structure(self):
        """Verify returns dict with all expected keys (total_clauses, high_risk_count, etc.)."""
        sample_clauses = [
            {"risk": {"level": "high"}},
            {"risk": {"level": "low"}}
        ]
        summary = calculate_risk_summary(sample_clauses)
        self.assertIn("total_clauses", summary)
        self.assertIn("high_risk_count", summary)
        self.assertIn("medium_risk_count", summary)
        self.assertIn("low_risk_count", summary)
        self.assertIn("risk_counts", summary)
        self.assertIn("overall_score", summary)
        self.assertIn("overall_level", summary)
        self.assertIn("overall_label", summary)

    def test_calculate_risk_summary_empty_list(self):
        """Verify handles empty clause list without errors."""
        summary = calculate_risk_summary([])
        self.assertEqual(summary["total_clauses"], 0)
        self.assertEqual(summary["high_risk_count"], 0)
        self.assertEqual(summary["medium_risk_count"], 0)
        self.assertEqual(summary["low_risk_count"], 0)
        self.assertEqual(summary["overall_level"], "low")

    def test_calculate_risk_summary_counts_correctly(self):
        """Verify risk level counts match input data exactly."""
        clauses = [
            {"risk": {"level": "high"}},
            {"risk": {"level": "high"}},
            {"risk": {"level": "medium"}},
            {"risk": {"level": "low"}},
            {"risk": {"level": "low"}},
            {"risk": {"level": "low"}}
        ]
        summary = calculate_risk_summary(clauses)
        self.assertEqual(summary["total_clauses"], 6)
        self.assertEqual(summary["high_risk_count"], 2)
        self.assertEqual(summary["medium_risk_count"], 1)
        self.assertEqual(summary["low_risk_count"], 3)
        self.assertEqual(summary["risk_counts"]["high"], 2)
        self.assertEqual(summary["risk_counts"]["medium"], 1)
        self.assertEqual(summary["risk_counts"]["low"], 3)

    def test_constants_defined(self):
        """Verify standard categories and risk levels are properly populated."""
        self.assertEqual(len(CLAUSE_CATEGORIES), 11)
        self.assertIn("Liability/Indemnification", CLAUSE_CATEGORIES)
        self.assertIn("Renewal/Auto-Renewal", CLAUSE_CATEGORIES)
        self.assertIn("low", RISK_LEVELS)
        self.assertIn("medium", RISK_LEVELS)
        self.assertIn("high", RISK_LEVELS)
        self.assertEqual(RISK_LEVELS["high"]["color"], "red")

    def test_extract_clauses(self):
        """Verify clause extraction correctly parses numbered sections."""
        sample_text = """1. PAYMENT TERMS\nClient shall pay invoices in 30 days.\n\n2. TERMINATION\nEither party may terminate on breach.\n"""
        clauses = extract_clauses(sample_text)
        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0]["title"], "Payment Terms")
        self.assertIn("Client shall pay", clauses[0]["text"])
        self.assertEqual(clauses[1]["title"], "Termination")

    def test_extract_clauses_fallback(self):
        """Verify unstructured paragraphs fall back to provision blocks."""
        unstructured = "First standalone paragraph here.\n\nSecond standalone paragraph here."
        clauses = extract_clauses(unstructured)
        self.assertEqual(len(clauses), 2)
        self.assertIn("First standalone", clauses[0]["text"])

    def test_classify_clause_categories(self):
        """Verify accurate categorization across legal domains."""
        self.assertEqual(classify_clause("Invoices are payable within 30 days of billing.", "PAYMENT"), "Payment Term")
        self.assertEqual(classify_clause("Either party may terminate this agreement upon notice.", "TERMINATION"), "Termination")
        self.assertEqual(classify_clause("Client shall indemnify, defend, and hold harmless vendor from claims.", "INDEMNITY"), "Liability/Indemnification")
        self.assertEqual(classify_clause("All trade secrets and proprietary data shall remain confidential.", "CONFIDENTIALITY"), "Confidentiality")
        self.assertEqual(classify_clause("Disputes shall be settled by binding arbitration before AAA.", "ARBITRATION"), "Dispute Resolution")
        self.assertEqual(classify_clause("This contract will automatically renew for successive terms unless non-renewal notice is given.", "RENEWAL"), "Renewal/Auto-Renewal")
        self.assertEqual(classify_clause("Late charges shall include a penalty fee and liquidated damages.", "PENALTY"), "Penalty")
        self.assertEqual(classify_clause("This agreement is governed by the laws of the State of Delaware.", "GOVERNING LAW"), "Governing Law")

    def test_assess_risk_high_flags(self):
        """Verify detection of high-risk red flags."""
        uncapped = "Client's liability shall be unlimited and shall not be subject to any cap."
        risk = assess_risk(uncapped, "Liability/Indemnification")
        self.assertEqual(risk["level"], "high")
        self.assertTrue(any("Uncapped liability" in r for r in risk["reasons"]))

        auto_renew = "This Agreement will automatically renew unless notice is given at least 90 days prior."
        risk = assess_risk(auto_renew, "Renewal/Auto-Renewal")
        self.assertEqual(risk["level"], "high")

        waiver = "CLIENT EXPRESSLY WAIVES ANY RIGHT TO A TRIAL BY JURY OR CLASS ACTION."
        risk = assess_risk(waiver, "Dispute Resolution")
        self.assertEqual(risk["level"], "high")

    def test_assess_risk_medium_and_low(self):
        """Verify detection of medium and low risk clauses."""
        med_text = "All fees must be paid within fifteen (15) days of invoice date."
        risk_med = assess_risk(med_text, "Payment Term")
        self.assertEqual(risk_med["level"], "medium")

        low_text = "The parties agree to communicate via email for general operational updates."
        risk_low = assess_risk(low_text, "Other")
        self.assertEqual(risk_low["level"], "low")

    def test_plain_english_summary(self):
        """Verify plain English translation returns relevant text."""
        risk = {"level": "high", "color": "red", "label": "High Risk"}
        summary = generate_plain_english_summary("Vendor limitation of liability", "Liability/Indemnification", risk)
        self.assertTrue(len(summary) > 20)
        self.assertTrue("financial responsibility" in summary.lower() or "damages" in summary.lower())

    def test_generate_lawyer_checklist(self):
        """Verify attorney preparation checklist generates actionable questions."""
        clauses = [
            {
                "number": "§5",
                "title": "Indemnification",
                "category": "Liability/Indemnification",
                "risk": {
                    "level": "high",
                    "reasons": ["One-sided indemnification"],
                    "recommendations": ["Require mutual indemnification"]
                }
            }
        ]
        checklist = generate_lawyer_checklist(clauses)
        self.assertTrue(len(checklist) >= 1)
        self.assertEqual(checklist[0]["priority"], "Urgent")
        self.assertIn("mutual indemnification", checklist[0]["action_item"].lower())

    @patch("analyzer.call_gemini_api", return_value=None)
    def test_analyze_document_full_pipeline(self, mock_api):
        """Verify end-to-end document analysis returns structured results."""
        doc_text = SAMPLE_CONTRACTS["vendor_sla"]["content"]
        result = analyze_document(doc_text)
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["metadata"]["total_clauses"], 8)
        self.assertGreaterEqual(result["metadata"]["risk_counts"]["high"], 2)
        self.assertEqual(result["metadata"]["overall_level"], "high")
        self.assertGreaterEqual(len(result["checklist"]), 2)
        self.assertIn("possible_next_steps", result)
        self.assertGreaterEqual(len(result["possible_next_steps"]), 2)


if __name__ == "__main__":
    unittest.main()
