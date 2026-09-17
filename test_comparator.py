"""Unit tests for ClauseWise legal document comparator module."""

import unittest
from unittest.mock import patch

from comparator import (
    build_comparison_prompt,
    calculate_comparison_metrics,
    compare_documents,
    compute_clause_similarity,
    generate_diff_html,
    DISCLAIMER_TEXT
)


class TestComparator(unittest.TestCase):
    """Test suite for legal document comparison functionality."""

    def test_build_comparison_prompt_contains_schema(self):
        """Verify prompt contains key schema elements (missing_in_a, differing_terms, etc.)."""
        prompt = build_comparison_prompt(
            doc_a_text="1. Term: 1 year.",
            doc_a_name="Version A",
            doc_b_text="1. Term: 2 years.",
            doc_b_name="Version B"
        )
        self.assertIn("missing_in_a", prompt)
        self.assertIn("missing_in_b", prompt)
        self.assertIn("differing_terms", prompt)
        self.assertIn("key_discussion_points", prompt)
        self.assertIn("doc_a_type", prompt)
        self.assertIn("documents_comparable", prompt)

    def test_build_comparison_prompt_neutral_instruction(self):
        """Verify prompt instructs against recommending which document to sign."""
        prompt = build_comparison_prompt(
            doc_a_text="Term A",
            doc_a_name="A",
            doc_b_text="Term B",
            doc_b_name="B"
        )
        self.assertIn("Do not recommend which document to sign", prompt)
        self.assertIn("NOT a lawyer", prompt)

    @patch("comparator.call_gemini_api", return_value=None)
    def test_compare_documents_handles_missing_file(self, mock_gemini):
        """Verify graceful error when a filepath doesn't exist."""
        result = compare_documents(
            filepath_a="non_existent_file_a_xyz.pdf",
            filename_a="Missing A",
            filepath_b="Some content for B",
            filename_b="B",
            api_key=""
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    def test_calculate_comparison_metrics_structure(self):
        """Verify returns dict with all expected keys."""
        sample_result = {
            "missing_in_a": [
                {"topic": "Audit Rights", "found_in_b": "Annual audit", "significance": "High"}
            ],
            "missing_in_b": [
                {"topic": "Data Retention", "found_in_a": "30 days", "significance": "Low"}
            ],
            "differing_terms": [
                {
                    "topic": "Payment Window",
                    "doc_a_says": "Net 30",
                    "doc_b_says": "Net 15",
                    "favors": "Favors vendor",
                    "significance": "High"
                }
            ]
        }
        metrics = calculate_comparison_metrics(sample_result)
        self.assertIn("total_differences", metrics)
        self.assertIn("missing_clauses_count", metrics)
        self.assertIn("differing_terms_count", metrics)
        self.assertIn("high_significance_count", metrics)
        self.assertEqual(metrics["total_differences"], 3)
        self.assertEqual(metrics["missing_clauses_count"], 2)
        self.assertEqual(metrics["differing_terms_count"], 1)
        self.assertEqual(metrics["high_significance_count"], 2)

    def test_calculate_comparison_metrics_empty(self):
        """Verify handles empty differences gracefully."""
        empty_metrics = calculate_comparison_metrics({})
        self.assertEqual(empty_metrics["total_differences"], 0)
        self.assertEqual(empty_metrics["missing_clauses_count"], 0)
        self.assertEqual(empty_metrics["differing_terms_count"], 0)
        self.assertEqual(empty_metrics["high_significance_count"], 0)

        none_metrics = calculate_comparison_metrics(None)
        self.assertEqual(none_metrics["total_differences"], 0)

    @patch("comparator.call_gemini_api", return_value=None)
    def test_compare_documents_pipeline(self, mock_gemini):
        """Verify full comparison pipeline returns structured JSON with disclaimer and metrics."""
        doc_a = "1. PAYMENT\nPayment is due within 30 days.\n\n2. CONFIDENTIALITY\nHeld confidential for 2 years."
        doc_b = "1. PAYMENT\nPayment is due within 15 days.\n\n3. INDEMNIFICATION\nClient shall indemnify vendor."
        result = compare_documents(
            filepath_a=doc_a,
            filename_a="Vendor Agreement v1",
            filepath_b=doc_b,
            filename_b="Vendor Agreement v2",
            api_key=""
        )
        self.assertEqual(result["status"], "success")
        self.assertIn("overall_comparison", result)
        self.assertIn("differing_terms", result)
        self.assertIn("missing_in_a", result)
        self.assertIn("missing_in_b", result)
        self.assertIn("metrics", result)
        self.assertEqual(result["disclaimer"], DISCLAIMER_TEXT)

    def test_compute_clause_similarity_and_diff(self):
        """Verify auxiliary clause similarity and HTML diff functions."""
        c1 = {"title": "Payment Terms", "number": "1", "text": "Client shall pay within 30 days."}
        c2 = {"title": "Payment Terms", "number": "1", "text": "Client shall pay within 15 days."}
        sim = compute_clause_similarity(c1, c2)
        self.assertGreater(sim, 0.6)

        diff_a, diff_b = generate_diff_html("Fee is $100", "Fee is $150")
        self.assertIn("<del class='diff-del'>", diff_a)
        self.assertIn("<ins class='diff-ins'>", diff_b)


if __name__ == "__main__":
    unittest.main()
