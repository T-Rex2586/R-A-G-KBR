#!/usr/bin/env python3
"""
Comprehensive Frontend, Security, and Compliance Verification Suite.
Validates:
- OWASP Top 10 (A01: Broken Access Control, A02: Cryptographic Failures, A03: Injection/XSS, A05: Security Misconfiguration)
- OWASP Top 10 for LLM Applications (LLM01: Prompt Injection, LLM02: Insecure Output Handling, LLM06: Sensitive Info Disclosure)
- OWASP ASVS v4.0 (V2: Authentication, V3: Session Management, V5: Validation & Sanitization, V14: Configuration)
- NIST CSF & ISO/IEC 27001/27002 Controls (Access Control, Cryptographic Protection, System Integrity)
- Frontend Unit & Component Integrity (DOM Sanitization, Accessibility, Theme Switching, Suggestion Chips)
"""

import os
import re
import json
import uuid
import sqlite3
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"
ASSETS_DIR = STATIC_DIR / "assets"

from unittest.mock import patch, MagicMock

@pytest.fixture(scope="module")
def client():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 384])

    mock_qdrant = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1109
    mock_qdrant.get_collection.return_value = mock_collection_info
    mock_query_response = MagicMock()
    mock_query_response.points = []
    mock_qdrant.query_points.return_value = mock_query_response

    test_db = PROJECT_ROOT / "app" / "test_sec_chat_storage.db"

    with patch("fastembed.TextEmbedding", return_value=mock_embedder), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant), \
         patch.dict(os.environ, {
             "GEMINI_API_KEY": "test_key_sec_12345",
             "QDRANT_HOST": "http://localhost:6333",
             "QDRANT_COLLECTION": "knowledge_base",
         }):
        with patch("app.server.DB_PATH", test_db):
            import app.server as server_module
            server_module.DB_PATH = test_db
            server_module.init_db()
            tc = TestClient(server_module.app)
            yield tc

        # Cleanup
        for f in [test_db, Path(str(test_db) + "-wal"), Path(str(test_db) + "-shm")]:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass


# =======================================================================
# 1. SECURITY TESTING (OWASP Top 10 & Sensitive Data Exposure)
# =======================================================================

class TestSecurityAudit:
    """Security audit against OWASP Top 10, ASVS, and ISO 27001 requirements."""

    def test_no_hardcoded_secrets_in_frontend_bundle(self):
        """OWASP A02 / ISO 27001: No API keys, passwords, or tokens hardcoded in client-side JS/HTML/CSS."""
        suspicious_patterns = [
            r'AIza[0-9A-Za-z-_]{35}',          # Google API key
            r'sk-[a-zA-Z0-9]{20,}',            # OpenAI / Generic secret key
            r'hf_[a-zA-Z0-9]{20,}',            # Hugging Face token
            r'redis_secure_pass',              # Redis password from .env
            r'n8n_secure_pass',                # Postgres password
        ]

        files_to_check = [
            STATIC_DIR / "index.html",
            ASSETS_DIR / "index-C1qRdULf.js",
            ASSETS_DIR / "index-BrPedg72.css"
        ]

        for file_path in files_to_check:
            assert file_path.exists(), f"File {file_path} missing!"
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in suspicious_patterns:
                matches = re.findall(pattern, content)
                assert not matches, f"Security Violation: Secret pattern '{pattern}' found in {file_path.name}: {matches}"

    def test_security_headers_enforcement(self, client):
        """OWASP A05 (Security Misconfiguration) & ASVS V14: HTTP Security Headers."""
        resp = client.get("/")
        assert resp.status_code == 200
        headers = resp.headers

        # X-Content-Type-Options
        assert headers.get("X-Content-Type-Options") == "nosniff"
        # Clickjacking defense (X-Frame-Options)
        assert headers.get("X-Frame-Options") == "DENY"
        # Referrer Policy
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        # Permissions Policy
        assert "camera=()" in headers.get("Permissions-Policy", "")
        assert "microphone=()" in headers.get("Permissions-Policy", "")

    def test_cookie_security_flags(self, client):
        """OWASP A01 / ASVS V3: Session cookies must have SameSite and HttpOnly flags."""
        import app.server as server_module
        fresh_client = TestClient(server_module.app)
        resp = fresh_client.get("/api/health")
        cookies = resp.headers.get_list("set-cookie")
        cookie_str = "; ".join(cookies)
        assert "sw_user_id=" in cookie_str
        assert "httponly" in cookie_str.lower()
        assert "samesite=lax" in cookie_str.lower()

    def test_xss_protection_in_frontend(self):
        """OWASP A03 / LLM02: Verify DOMPurify or equivalent sanitization is enforced before rendering markdown."""
        js_content = (ASSETS_DIR / "index-C1qRdULf.js").read_text(encoding="utf-8", errors="ignore")
        # Ensure sanitize is called on parsed markdown
        assert "sanitize" in js_content, "DOMPurify / sanitize must be integrated into markdown rendering pipeline"

    def test_anti_hallucination_guardrail_present(self):
        """NIST AI RMF / OWASP LLM01: System prompt must include strict anti-hallucination out-of-domain instruction."""
        from app.server import SYSTEM_INSTRUCTION
        assert "Aturan Anti-Halusinasi" in SYSTEM_INSTRUCTION
        assert "Maaf, informasi mengenai hal tersebut tidak ditemukan" in SYSTEM_INSTRUCTION


# =======================================================================
# 2. FRONTEND INTEGRITY & COMPONENT TESTING
# =======================================================================

class TestFrontendIntegrity:
    """Tests frontend assets, accessibility attributes, theme handling, and suggestions."""

    def test_index_html_structure_and_meta(self):
        """Accessibility & SEO: HTML5 semantics, meta viewport, title, and lang attribute."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "<!doctype html>" in html.lower()
        assert 'lang="id"' in html
        assert 'name="viewport"' in html
        assert 'name="description"' in html
        assert "<title>KBR Bot" in html

    def test_theme_anti_flash_script_present(self):
        """UI/UX: Prevents flash of unstyled theme on page load."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "sw_rag_theme" in html
        assert "data-theme" in html
        assert "localStorage.getItem" in html

    def test_interactive_followup_suggestions_css_and_js(self):
        """UI Improvement: Interactive follow-up suggestion chips (↳) styling and event delegation."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert ".rag-suggestion-chip" in html
        assert ".rag-suggestion-arrow" in html
        assert "enhanceFollowUpSuggestions" in html
        assert "data-question" in html
        assert ".chat-textarea" in html
        assert ".btn-send" in html

    def test_desktop_sidebar_toggle_css(self):
        """UI Improvement: Desktop sidebar collapse & expand styling."""
        css = (ASSETS_DIR / "index-BrPedg72.css").read_text(encoding="utf-8")
        assert "@media (min-width: 769px)" in css or "@media (min-width:769px)" in css
        assert ".sidebar.open" in css
        assert "margin-left" in css

    def test_light_and_dark_mode_css_variables(self):
        """Light & Dark mode variable definitions."""
        css = (ASSETS_DIR / "index-BrPedg72.css").read_text(encoding="utf-8")
        # Check root or light mode variables
        assert "--bg-canvas" in css
        assert "--text-primary" in css
        assert "--accent-primary" in css
        # Check dark mode selector
        assert '[data-theme="dark"]' in css or '[data-theme=dark]' in css


# =======================================================================
# 3. END-TO-END API & COMPONENT INTEGRATION TESTING
# =======================================================================

class TestAPIIntegration:
    """Integration testing: Session creation, messaging, retrieval, and deletion."""

    def test_health_check_payload_clean(self, client):
        """Health endpoint should expose points count without leaking environment keys."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "healthy"
        assert "points_count" in data
        assert "GEMINI_API_KEY" not in data
        assert "QDRANT_API_KEY" not in data

    def test_chat_creates_session_and_persists_history(self, client):
        """User can send a chat message and fetch session details."""
        user_headers = {"X-User-ID": "usr_test_integration_user"}
        resp = client.post(
            "/api/chat",
            json={"message": "Apa itu Ontologi dalam Semantic Web?"},
            headers=user_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        session_id = data["session_id"]
        assert session_id is not None

        # Verify session is listed for this user
        list_resp = client.get("/api/sessions", headers=user_headers)
        assert list_resp.status_code == 200
        sessions = list_resp.json().get("sessions", [])
        assert any(s["id"] == session_id for s in sessions)

        # Verify session details can be retrieved
        detail_resp = client.get(f"/api/sessions/{session_id}", headers=user_headers)
        assert detail_resp.status_code == 200
        messages = detail_resp.json().get("messages", [])
        assert len(messages) >= 2  # User message + Model reply

    def test_multi_user_isolation(self, client):
        """OWASP A01: User A cannot see or access sessions belonging to User B."""
        headers_a = {"X-User-ID": "usr_alice_123"}
        headers_b = {"X-User-ID": "usr_bob_456"}

        # Alice creates a chat
        resp_a = client.post(
            "/api/chat",
            json={"message": "Catatan rahasia Alice mengenai RDF."},
            headers=headers_a
        )
        alice_session_id = resp_a.json()["session_id"]

        # Bob lists sessions — should NOT contain Alice's session
        resp_b_list = client.get("/api/sessions", headers=headers_b)
        bob_session_ids = [s["id"] for s in resp_b_list.json().get("sessions", [])]
        assert alice_session_id not in bob_session_ids

        # Bob attempts direct access to Alice's session ID — must return empty/not found
        resp_b_detail = client.get(f"/api/sessions/{alice_session_id}", headers=headers_b)
        assert len(resp_b_detail.json().get("messages", [])) == 0

    def test_clear_all_clears_only_own_sessions(self, client):
        """Clearing sessions only wipes the caller's data, preserving other users."""
        uid_x = f"usr_x_{uuid.uuid4().hex[:8]}"
        uid_y = f"usr_y_{uuid.uuid4().hex[:8]}"
        headers_x = {"X-User-ID": uid_x}
        headers_y = {"X-User-ID": uid_y}

        client.post("/api/chat", json={"message": "Halo dari X"}, headers=headers_x)
        client.post("/api/chat", json={"message": "Halo dari Y"}, headers=headers_y)

        # X clears all sessions
        clear_resp = client.post("/api/clear", headers=headers_x)
        assert clear_resp.status_code == 200

        # X now has 0 sessions
        assert len(client.get("/api/sessions", headers=headers_x).json()["sessions"]) == 0

        # Y still has their session!
        assert len(client.get("/api/sessions", headers=headers_y).json()["sessions"]) == 1
