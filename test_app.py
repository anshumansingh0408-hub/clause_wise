"""Integration and route tests for ClauseWise Flask application."""

import unittest
from unittest.mock import patch
import io
from app import app, RATE_LIMIT_TRACKER


class TestApp(unittest.TestCase):
    """Test suite for ClauseWise Flask application routes and REST API."""

    def setUp(self):
        """Configure test client and reset rate limit counters."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        RATE_LIMIT_TRACKER.clear()

    def test_index_route(self):
        """GET / returns 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ClauseWise", response.data)

    def test_about_route(self):
        """GET /about returns 200."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)

    def test_security_headers(self):
        """GET / has X-Frame-Options and CSP headers."""
        response = self.client.get("/")
        self.assertIn("X-Frame-Options", response.headers)
        self.assertIn("Content-Security-Policy", response.headers)

    def test_analyze_no_file(self):
        """POST /api/analyze with no file returns 400."""
        response = self.client.post("/api/analyze")
        self.assertEqual(response.status_code, 400)

    def test_analyze_invalid_extension(self):
        """POST /api/analyze with .exe file returns 400."""
        data = {
            "file": (io.BytesIO(b"executable content"), "malicious.exe")
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    @patch("app.analyze_document")
    def test_analyze_valid_pdf(self, mock_analyze):
        """Mock analyze_document to return a fixed result; POST with valid file returns 200."""
        mock_analyze.return_value = {
            "status": "success",
            "metadata": {
                "total_clauses": 1,
                "total_words": 20,
                "analysis_time_sec": 0.05,
                "overall_score": 10,
                "overall_level": "low",
                "overall_label": "Low Risk",
                "risk_counts": {"high": 0, "medium": 0, "low": 1}
            },
            "clauses": [
                {
                    "id": "clause-1",
                    "number": "§1",
                    "title": "Term",
                    "category": "Termination",
                    "text": "This agreement shall remain effective for 1 year.",
                    "word_count": 8,
                    "risk": {"level": "low", "color": "green", "label": "Standard Clause"},
                    "plain_english": "1 year agreement term."
                }
            ],
            "checklist": []
        }

        data = {
            "file": (io.BytesIO(b"%PDF-1.4 mock pdf text content"), "agreement.pdf")
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("job_id", json_data)
        self.assertIn("result", json_data)
        self.assertIn("risk_summary", json_data)
        self.assertIn("disclaimer", json_data)

    def test_compare_missing_one_file(self):
        """POST /api/compare with only file_a returns 400."""
        data = {
            "file_a": (io.BytesIO(b"Only doc a content"), "doc_a.txt")
        }
        response = self.client.post("/api/compare", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_ask_unknown_job(self):
        """POST /api/ask with nonexistent job_id returns 404."""
        response = self.client.post("/api/ask", json={
            "job_id": "nonexistent-job-uuid-12345",
            "question": "What is the liability cap?"
        })
        self.assertEqual(response.status_code, 404)

    def test_ask_empty_question(self):
        """POST /api/ask with empty question returns 400."""
        response = self.client.post("/api/ask", json={
            "job_id": "any-job-id",
            "question": ""
        })
        self.assertEqual(response.status_code, 400)

    def test_status_unknown_job(self):
        """GET /api/status/nonexistent returns 404."""
        response = self.client.get("/api/status/nonexistent-job-id-999")
        self.assertEqual(response.status_code, 404)

    def test_sample_endpoint(self):
        """GET /api/sample returns 200 with JSON."""
        response = self.client.get("/api/sample")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        json_data = response.get_json()
        self.assertIn("risky_lease_clause", json_data)

    def test_response_time_header(self):
        """Verify X-Response-Time header present."""
        response = self.client.get("/")
        self.assertIn("X-Response-Time", response.headers)

    def test_rate_limiting(self):
        """POST /api/analyze 16 times rapidly, verify 16th request returns 429."""
        RATE_LIMIT_TRACKER.clear()
        for _ in range(15):
            res = self.client.post("/api/analyze")
            self.assertEqual(res.status_code, 400)

        res16 = self.client.post("/api/analyze")
        self.assertEqual(res16.status_code, 429)

    # Additional server-rendered template & helper route tests
    def test_analyze_get_default_sample(self):
        """Test GET /analyze defaults to sample analysis."""
        response = self.client.get("/analyze")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"clauses analyzed", response.data)

    def test_analyze_get_specific_sample(self):
        """Test GET /analyze with query param loads target sample."""
        response = self.client.get("/analyze?sample=mutual_nda")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Mutual Non-Disclosure Agreement", response.data)

    def test_analyze_post_text(self):
        """Test POST /analyze with raw text."""
        payload = {
            "document_title": "Test Agreement",
            "document_text": "1. PAYMENT\nInvoices must be paid in 30 days.\n\n2. TERMINATION\nNotice of 30 days."
        }
        response = self.client.post("/analyze", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Test Agreement", response.data)

    def test_analyze_post_empty_redirects(self):
        """Test POST /analyze without text redirects to home."""
        response = self.client.post("/analyze", data={"document_text": ""})
        self.assertEqual(response.status_code, 302)

    def test_compare_get_page(self):
        """Test GET /compare loads setup form."""
        response = self.client.get("/compare")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Version A", response.data)

    def test_compare_get_demo(self):
        """Test GET /compare?pair=demo executes comparison."""
        response = self.client.get("/compare?pair=demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Comparing", response.data)

    def test_compare_post_documents(self):
        """Test POST /compare runs diff and returns results."""
        data = {
            "title_a": "Draft 1",
            "text_a": "1. LIABILITY\nLiability capped at $1000.",
            "title_b": "Draft 2",
            "text_b": "1. LIABILITY\nClient liability is unlimited."
        }
        response = self.client.post("/compare", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Draft 1", response.data)
        self.assertIn(b"Draft 2", response.data)

    def test_about_page_content(self):
        """Test methodology and disclaimer content in about page."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Methodology & Legal Standards", response.data)
        self.assertIn(b"Obligation", response.data)

    def test_api_sample_by_key(self):
        """Test sample fetching API by key."""
        response = self.client.get("/api/sample/vendor_sla")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("content", data)
        self.assertIn("MASTER SERVICES AGREEMENT", data["content"])

    def test_api_export_checklist(self):
        """Test checklist Markdown export."""
        payload = {
            "title": "Vendor MSA",
            "items": [
                {
                    "clause_number": "§5",
                    "clause_title": "Indemnification",
                    "category": "Liability/Indemnification",
                    "priority": "Urgent",
                    "concern": "Unilateral indemnification",
                    "action_item": "Request mutual indemnification"
                }
            ]
        }
        response = self.client.post("/api/export-checklist", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"# ClauseWise Lawyer-Prep Checklist", response.data)

    def test_health_check(self):
        """Test health check route."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
