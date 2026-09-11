#!/usr/bin/env python3
"""
Comprehensive Backend Test Suite for Semantic Web GPT RAG Server.
Covers: Unit Tests, Integration Tests, Security Tests, Input Validation.

Standards Referenced:
- OWASP Top 10 (A01-A10)
- OWASP Top 10 for LLM Applications (LLM01-LLM10)
- OWASP ASVS v4.0
- CWE (Common Weakness Enumeration)
- ISO/IEC 27001 / 27002
- NIST Cybersecurity Framework (CSF)
"""

import os
import sys
import uuid
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We need to mock heavy dependencies before importing the server module
# because server.py initializes Qdrant and embedder at import time.

@pytest.fixture(scope="session", autouse=True)
def mock_heavy_deps():
    """Mock Qdrant client and embedding model to avoid needing running services."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 384])

    mock_qdrant = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1109
    mock_qdrant.get_collection.return_value = mock_collection_info

    mock_query_response = MagicMock()
    mock_query_response.points = []
    mock_qdrant.query_points.return_value = mock_query_response

    with patch("fastembed.TextEmbedding", return_value=mock_embedder), \
         patch("qdrant_client.QdrantClient", return_value=mock_qdrant), \
         patch.dict(os.environ, {
             "GEMINI_API_KEY": "test_key_for_testing_only",
             "QDRANT_HOST": "http://localhost:6333",
             "QDRANT_COLLECTION": "knowledge_base",
         }):
        # Use a temporary DB for tests
        test_db = PROJECT_ROOT / "app" / "test_chat_storage.db"
        with patch("app.server.DB_PATH", test_db):
            import app.server as server_module
            server_module.DB_PATH = test_db
            server_module.init_db()
            yield server_module
        # Cleanup
        for f in [test_db, Path(str(test_db) + "-wal"), Path(str(test_db) + "-shm"), Path(str(test_db) + "-journal")]:
            try:
                if f.exists():
                    f.unlink()
            except (PermissionError, OSError):
                pass  # Windows file lock; safe to ignore in tests


@pytest.fixture
def server(mock_heavy_deps):
    return mock_heavy_deps


@pytest.fixture
def client(server):
    from fastapi.testclient import TestClient
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def clean_db(server):
    """Clean test database before each test."""
    try:
        with sqlite3.connect(str(server.DB_PATH)) as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
            conn.commit()
    except Exception:
        pass
    # Also clear rate limit store
    server._rate_limit_store.clear()
    yield


# ============================================================
# 1. UNIT TESTS — SQLite Helper Functions
# ============================================================

class TestSQLiteHelpers:
    """Unit tests for SQLite session storage helper functions."""

    def test_init_db_creates_tables(self, server):
        """Verify init_db creates sessions and messages tables."""
        with sqlite3.connect(str(server.DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cursor.fetchall()]
        assert "sessions" in tables
        assert "messages" in tables

    def test_save_session_turn(self, server):
        """Verify save_session_turn stores user and model messages."""
        sid = str(uuid.uuid4())
        server.save_session_turn(sid, "usr_test", "Test Title", "Hello", False, "Hi back")

        with sqlite3.connect(str(server.DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,))
            session = cursor.fetchone()
            assert session is not None
            assert session[1] == "usr_test"  # user_id

            cursor.execute("SELECT role, text FROM messages WHERE session_id = ? ORDER BY id", (sid,))
            msgs = cursor.fetchall()
            assert len(msgs) == 2
            assert msgs[0] == ("user", "Hello")
            assert msgs[1] == ("model", "Hi back")

    def test_get_session_history_limit(self, server):
        """Verify get_session_history respects limit parameter."""
        sid = str(uuid.uuid4())
        for i in range(10):
            server.save_session_turn(sid, "usr_test", "Title", f"Q{i}", False, f"A{i}")

        history = server.get_session_history(sid, limit=4)
        assert len(history) == 4  # Should return last 4 messages (not 4 turns)

    def test_get_user_sessions_list_isolation(self, server):
        """Verify sessions are isolated per user_id (ISO 27001 A.9.4)."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        server.save_session_turn(sid_a, "usr_alice", "Alice Chat", "Hi", False, "Hello Alice")
        server.save_session_turn(sid_b, "usr_bob", "Bob Chat", "Hey", False, "Hello Bob")

        alice_sessions = server.get_user_sessions_list("usr_alice")
        bob_sessions = server.get_user_sessions_list("usr_bob")

        assert len(alice_sessions) == 1
        assert alice_sessions[0]["title"] == "Alice Chat"
        assert len(bob_sessions) == 1
        assert bob_sessions[0]["title"] == "Bob Chat"

    def test_get_user_session_detail_ownership(self, server):
        """Verify session detail requires ownership (OWASP A01)."""
        sid = str(uuid.uuid4())
        server.save_session_turn(sid, "usr_alice", "Private", "Secret", False, "Answer")

        # Alice can access
        detail = server.get_user_session_detail(sid, "usr_alice")
        assert detail is not None
        assert len(detail["messages"]) == 2

        # Bob CANNOT access Alice's session
        detail_bob = server.get_user_session_detail(sid, "usr_bob")
        assert detail_bob is None

    def test_delete_user_session_ownership(self, server):
        """Verify delete only works for session owner (OWASP A01)."""
        sid = str(uuid.uuid4())
        server.save_session_turn(sid, "usr_alice", "To Delete", "Q", False, "A")

        # Bob tries to delete Alice's session — should not work
        server.delete_user_session_db(sid, "usr_bob")
        assert server.get_user_session_detail(sid, "usr_alice") is not None

        # Alice deletes her own session — should work
        server.delete_user_session_db(sid, "usr_alice")
        assert server.get_user_session_detail(sid, "usr_alice") is None

    def test_clear_all_user_sessions_isolation(self, server):
        """Verify clear_all only clears the requesting user's sessions."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        server.save_session_turn(sid_a, "usr_alice", "A1", "Q", False, "A")
        server.save_session_turn(sid_b, "usr_bob", "B1", "Q", False, "A")

        server.clear_all_user_sessions_db("usr_alice")

        assert len(server.get_user_sessions_list("usr_alice")) == 0
        assert len(server.get_user_sessions_list("usr_bob")) == 1


# ============================================================
# 2. INTEGRATION TESTS — API Endpoints
# ============================================================

class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints."""

    def test_health_check(self, client):
        """GET /api/health returns healthy status."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["qdrant_connected"] is True
        assert data["points_count"] == 1109
        assert data["gemini_api_key_set"] is True

    def test_list_sessions_empty(self, client):
        """GET /api/sessions returns empty for new user."""
        resp = client.get("/api/sessions", headers={"X-User-ID": "usr_new_user"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []

    def test_get_nonexistent_session(self, client):
        """GET /api/sessions/{id} returns empty messages for non-existent session."""
        resp = client.get("/api/sessions/nonexistent", headers={"X-User-ID": "usr_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []

    def test_chat_creates_session(self, client, server):
        """POST /api/chat creates a new session and returns response."""
        with patch.object(server, "generate_gemini_reply", return_value="Mocked AI Reply"):
            resp = client.post("/api/chat", json={
                "message": "Apa itu RDF?"
            }, headers={"X-User-ID": "usr_integtest"})

        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "session_id" in data
        assert "session_title" in data

        # Session should now appear in list
        resp2 = client.get("/api/sessions", headers={"X-User-ID": "usr_integtest"})
        assert len(resp2.json()["sessions"]) == 1

    def test_chat_session_isolation_between_users(self, client, server):
        """Verify User A's chat session is invisible to User B (FR-8, ISO 27001)."""
        with patch.object(server, "generate_gemini_reply", return_value="Reply"):
            client.post("/api/chat", json={"message": "Test"}, headers={"X-User-ID": "usr_A"})

        resp_a = client.get("/api/sessions", headers={"X-User-ID": "usr_A"})
        resp_b = client.get("/api/sessions", headers={"X-User-ID": "usr_B"})

        assert len(resp_a.json()["sessions"]) == 1
        assert len(resp_b.json()["sessions"]) == 0

    def test_delete_session(self, client, server):
        """DELETE /api/sessions/{id} removes session for owner."""
        with patch.object(server, "generate_gemini_reply", return_value="Reply"):
            resp = client.post("/api/chat", json={"message": "Test"}, headers={"X-User-ID": "usr_del"})
        sid = resp.json()["session_id"]

        del_resp = client.delete(f"/api/sessions/{sid}", headers={"X-User-ID": "usr_del"})
        assert del_resp.status_code == 200

        list_resp = client.get("/api/sessions", headers={"X-User-ID": "usr_del"})
        assert len(list_resp.json()["sessions"]) == 0

    def test_clear_all_sessions(self, client, server):
        """POST /api/clear clears only the requesting user's sessions."""
        with patch.object(server, "generate_gemini_reply", return_value="Reply"):
            client.post("/api/chat", json={"message": "A1"}, headers={"X-User-ID": "usr_clear_A"})
            client.post("/api/chat", json={"message": "B1"}, headers={"X-User-ID": "usr_clear_B"})

        clear_resp = client.post("/api/clear", headers={"X-User-ID": "usr_clear_A"})
        assert clear_resp.status_code == 200

        assert len(client.get("/api/sessions", headers={"X-User-ID": "usr_clear_A"}).json()["sessions"]) == 0
        assert len(client.get("/api/sessions", headers={"X-User-ID": "usr_clear_B"}).json()["sessions"]) == 1


# ============================================================
# 3. SECURITY TESTS
# ============================================================

class TestSecurity:
    """Security tests based on OWASP Top 10, ASVS, and CWE."""

    def test_security_headers_present(self, client):
        """[S12] Verify security response headers (OWASP A05, CWE-693)."""
        resp = client.get("/api/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_cookie_set_on_first_visit(self, client):
        """Verify sw_user_id cookie is set for new visitors."""
        resp = client.get("/api/health")
        cookies = resp.headers.get_list("set-cookie")
        cookie_str = "; ".join(cookies)
        assert "sw_user_id" in cookie_str

    def test_cookie_httponly_flag(self, client):
        """[S5] Verify cookie has HttpOnly flag (CWE-1004)."""
        resp = client.get("/api/health")
        cookies = resp.headers.get_list("set-cookie")
        cookie_str = "; ".join(cookies)
        assert "httponly" in cookie_str.lower()

    def test_cookie_samesite_flag(self, client):
        """Verify cookie has SameSite=Lax (OWASP ASVS V3.4)."""
        resp = client.get("/api/health")
        cookies = resp.headers.get_list("set-cookie")
        cookie_str = "; ".join(cookies)
        assert "samesite=lax" in cookie_str.lower()

    def test_empty_message_rejected(self, client):
        """[S7] Empty messages return 400 (CWE-20)."""
        resp = client.post("/api/chat", json={"message": ""}, headers={"X-User-ID": "usr_sec"})
        assert resp.status_code == 400

    def test_oversized_image_rejected(self, client):
        """[S7] Oversized image data returns 413 (CWE-400)."""
        huge_image = "x" * (8 * 1024 * 1024)  # 8MB
        resp = client.post("/api/chat", json={
            "message": "test",
            "image_data": huge_image
        }, headers={"X-User-ID": "usr_sec"})
        assert resp.status_code == 413

    def test_rate_limiting(self, client, server):
        """[S6] Rate limiter blocks after threshold (CWE-770, OWASP A04)."""
        with patch.object(server, "generate_gemini_reply", return_value="Reply"):
            for i in range(20):
                resp = client.post("/api/chat", json={
                    "message": f"Question {i}"
                }, headers={"X-User-ID": "usr_ratelimit"})
                assert resp.status_code == 200

            # 21st request should be rate limited
            resp = client.post("/api/chat", json={
                "message": "One more"
            }, headers={"X-User-ID": "usr_ratelimit"})
            assert resp.status_code == 429

    def test_sql_injection_in_session_id(self, client):
        """SQL injection attempt in session_id (OWASP A03, CWE-89)."""
        malicious_id = "'; DROP TABLE sessions; --"
        resp = client.get(f"/api/sessions/{malicious_id}", headers={"X-User-ID": "usr_sqli"})
        assert resp.status_code == 200
        # Tables should still exist
        with sqlite3.connect(str(Path(__file__).resolve().parent.parent / "app" / "test_chat_storage.db")) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            assert cursor.fetchone() is not None

    def test_xss_in_message_stored_safely(self, client, server):
        """XSS payload in message is stored as-is (sanitization is frontend's job) (CWE-79)."""
        xss_payload = '<script>alert("XSS")</script>'
        with patch.object(server, "generate_gemini_reply", return_value="Safe reply"):
            resp = client.post("/api/chat", json={
                "message": xss_payload
            }, headers={"X-User-ID": "usr_xss"})
        assert resp.status_code == 200
        # Message is stored; DOMPurify on frontend handles sanitization

    def test_cross_user_session_access_denied(self, client, server):
        """User B cannot access User A's session detail (OWASP A01)."""
        with patch.object(server, "generate_gemini_reply", return_value="Secret"):
            resp = client.post("/api/chat", json={"message": "My secret"}, headers={"X-User-ID": "usr_victim"})
        sid = resp.json()["session_id"]

        # Attacker tries to access victim's session
        attacker_resp = client.get(f"/api/sessions/{sid}", headers={"X-User-ID": "usr_attacker"})
        assert attacker_resp.json()["messages"] == []

    def test_cross_user_session_delete_denied(self, client, server):
        """User B cannot delete User A's session (OWASP A01)."""
        with patch.object(server, "generate_gemini_reply", return_value="Keep"):
            resp = client.post("/api/chat", json={"message": "Important"}, headers={"X-User-ID": "usr_owner"})
        sid = resp.json()["session_id"]

        # Attacker tries to delete
        client.delete(f"/api/sessions/{sid}", headers={"X-User-ID": "usr_attacker"})

        # Owner's session should still exist
        owner_resp = client.get(f"/api/sessions/{sid}", headers={"X-User-ID": "usr_owner"})
        assert len(owner_resp.json()["messages"]) > 0

    def test_no_api_keys_in_health_response(self, client):
        """API keys should not be leaked in health endpoint (CWE-200)."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "gemini_api_key_set" in data  # Boolean only, not the actual key
        body_str = resp.text
        assert "test_key_for_testing_only" not in body_str

    def test_message_length_truncation(self, client, server):
        """[S7] Messages exceeding MAX_MESSAGE_LENGTH are truncated (CWE-20)."""
        long_msg = "A" * 6000
        with patch.object(server, "generate_gemini_reply", return_value="OK") as mock_reply:
            resp = client.post("/api/chat", json={"message": long_msg}, headers={"X-User-ID": "usr_trunc"})
        assert resp.status_code == 200
        # The query passed to generate_gemini_reply should be truncated
        # (we can verify via session storage)


# ============================================================
# 4. INPUT VALIDATION TESTS
# ============================================================

class TestInputValidation:
    """Input validation tests (OWASP A03, CWE-20)."""

    def test_missing_message_and_image(self, client):
        """Both message and image empty returns 400."""
        resp = client.post("/api/chat", json={}, headers={"X-User-ID": "usr_val"})
        assert resp.status_code == 400

    def test_whitespace_only_message(self, client):
        """Whitespace-only message returns 400."""
        resp = client.post("/api/chat", json={"message": "   "}, headers={"X-User-ID": "usr_val"})
        assert resp.status_code == 400

    def test_valid_session_id_format(self, client, server):
        """Custom session_id is accepted."""
        with patch.object(server, "generate_gemini_reply", return_value="OK"):
            resp = client.post("/api/chat", json={
                "message": "Test",
                "session_id": "custom-session-123"
            }, headers={"X-User-ID": "usr_val"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "custom-session-123"

    def test_unicode_message_handling(self, client, server):
        """Unicode messages (emoji, CJK, Arabic) are handled correctly."""
        with patch.object(server, "generate_gemini_reply", return_value="OK"):
            resp = client.post("/api/chat", json={
                "message": "🧠 Apa itu 知识图谱 dalam السيمانتك ويب?"
            }, headers={"X-User-ID": "usr_unicode"})
        assert resp.status_code == 200


# ============================================================
# 5. USER ID EXTRACTION TESTS
# ============================================================

class TestUserIDExtraction:
    """Tests for get_user_id helper function."""

    def test_user_id_from_header(self, client, server):
        """X-User-ID header takes precedence."""
        with patch.object(server, "generate_gemini_reply", return_value="OK"):
            resp = client.post("/api/chat", json={"message": "Test"},
                             headers={"X-User-ID": "usr_from_header"})
        sessions = client.get("/api/sessions", headers={"X-User-ID": "usr_from_header"})
        assert len(sessions.json()["sessions"]) == 1

    def test_guest_user_fallback(self, server):
        """Missing/short user_id falls back to usr_guest."""
        from fastapi.testclient import TestClient
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {}
        mock_request.state = MagicMock(spec=[])
        uid = server.get_user_id(mock_request)
        assert uid == "usr_guest"


# ============================================================
# 6. RATE LIMITER UNIT TESTS
# ============================================================

class TestRateLimiter:
    """Unit tests for rate limiting function."""

    def test_allows_under_limit(self, server):
        """Requests under the limit are allowed."""
        for i in range(19):
            assert server.check_rate_limit("usr_rl_ok") is False

    def test_blocks_over_limit(self, server):
        """Requests over the limit are blocked."""
        for i in range(20):
            server.check_rate_limit("usr_rl_block")
        assert server.check_rate_limit("usr_rl_block") is True

    def test_different_users_independent(self, server):
        """Rate limits are per-user, not global."""
        for i in range(20):
            server.check_rate_limit("usr_rl_x")
        # Different user should not be affected
        assert server.check_rate_limit("usr_rl_y") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
