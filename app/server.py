#!/usr/bin/env python3
"""
RAG ChatGPT Web Application Server
Menghubungkan antarmuka web ChatGPT dengan Qdrant Vector Store (12 Modul Kuliah)
dan Google Gemini 2.5 Flash dengan Multi-Turn Session Memory dan anti-halusinasi ketat.
"""

import os
import sys
import uuid
import logging
import time
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from dotenv import load_dotenv
import sqlite3
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Plus

# Load environment
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-gpt-server")

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
raw_model = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")
GEMINI_MODEL = raw_model.replace("models/", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")

# Initialize Vector Search & Embedder
logger.info("Memuat model embedding sentence-transformers/all-MiniLM-L6-v2...")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

logger.info("Memuat model Cross-Encoder Reranker FlashRank (ms-marco-TinyBERT-L-2-v2)...")
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

logger.info(f"Menghubungkan ke Qdrant di {QDRANT_HOST}...")
try:
    qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=10)
    logger.info("✓ Koneksi Qdrant berhasil")
except Exception as e:
    logger.warning(f"⚠ Qdrant connection issue: {e}")
    qdrant_client = None

# SQLite Persistent Session Storage dengan Isolasi User ID
DB_PATH = BASE_DIR / "app" / "chat_storage.db"

def init_db():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                has_image INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
            conn.commit()
    except Exception as e:
        logger.error(f"Inisialisasi SQLite error: {e}")

init_db()

def get_user_id(request: Request) -> str:
    """Mengambil user_id unik per client dari header atau cookie."""
    uid = request.headers.get("x-user-id")
    if not uid:
        uid = request.cookies.get("sw_user_id")
    if not uid:
        uid = getattr(request.state, "user_id", None)
    if not uid or len(uid.strip()) < 3:
        uid = "usr_guest"
    return uid.strip()

def get_session_history(session_id: str, limit: int = 8) -> List[Dict[str, any]]:
    """Mengambil percakapan sebelumnya untuk konteks model."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, text, has_image FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": r[0], "text": r[1], "has_image": bool(r[2])} for r in reversed(rows)]
    except Exception as e:
        logger.error(f"Error get_session_history: {e}")
        return []

def save_session_turn(session_id: str, user_id: str, title: str, user_msg: str, has_image: bool, bot_reply: str):
    """Menyimpan turn percakapan user dan bot ke sesi milik user_id."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (session_id, user_id, title, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, user_id, title)
            )
            cursor.execute(
                "INSERT INTO messages (session_id, role, text, has_image) VALUES (?, 'user', ?, ?)",
                (session_id, user_msg, 1 if has_image else 0)
            )
            cursor.execute(
                "INSERT INTO messages (session_id, role, text, has_image) VALUES (?, 'model', ?, 0)",
                (session_id, bot_reply)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error save_session_turn: {e}")

def get_user_sessions_list(user_id: str) -> List[Dict[str, any]]:
    """Mengambil daftar sesi yang HANYA milik user_id ini."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.session_id, s.title,
                       (SELECT COUNT(*) FROM messages WHERE session_id = s.session_id) as msg_count,
                       (SELECT text FROM messages WHERE session_id = s.session_id ORDER BY id DESC LIMIT 1) as preview
                FROM sessions s
                WHERE s.user_id = ?
                ORDER BY s.updated_at DESC
            """, (user_id,))
            rows = cursor.fetchall()
            sessions = []
            for r in rows:
                sessions.append({
                    "id": r[0],
                    "title": r[1] or "Percakapan",
                    "message_count": r[2] or 0,
                    "preview": (r[3] or "")[:60]
                })
            return sessions
    except Exception as e:
        logger.error(f"Error get_user_sessions_list: {e}")
        return []

def get_user_session_detail(session_id: str, user_id: str) -> Optional[Dict[str, any]]:
    """Mengambil pesan dalam sesi HANYA jika sesi tersebut dimiliki oleh user_id."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
            row = cursor.fetchone()
            if not row:
                return None
            title = row[0]
            cursor.execute("SELECT role, text, has_image FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
            rows = cursor.fetchall()
            messages = [{"role": r[0], "text": r[1], "has_image": bool(r[2])} for r in rows]
            return {
                "session_id": session_id,
                "title": title,
                "messages": messages
            }
    except Exception as e:
        logger.error(f"Error get_user_session_detail: {e}")
        return None

def delete_user_session_db(session_id: str, user_id: str) -> bool:
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ? AND session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)", (session_id, user_id))
            cursor.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error delete_user_session_db: {e}")
        return False

def clear_all_user_sessions_db(user_id: str) -> bool:
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)", (user_id,))
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error clear_all_user_sessions_db: {e}")
        return False

SYSTEM_INSTRUCTION = """Anda adalah Asisten AI Spesialis untuk materi perkuliahan Semantic Web & Knowledge Engineering.
Materi perkuliahan mencakup 12 modul utama:
- 0.0 Overview & 0.1 Introduction
- 1. Riset Semantic Web
- 2.0 SWTech & 2.1 AP-RDF (Resource Description Framework, RDF Triple: Subject-Predicate-Object, Graf RDF, RDFS)
- 3.0 SWTech - Ontology & 3.1 Protege (Ontologi, Kelas, Properti/Relasi, Individu, Alat Protege)
- 4. Ontology Engineering & 5. Knowledge Base (TBox/terminologi, ABox/fakta assertions, RBox)
- 6. SPARQL (Query language graf RDF: SELECT, WHERE, ASK, CONSTRUCT, FILTER)
- 7. Reasoning & 8. SWRL (Semantic Web Rule Language, Inferensi logika, Antecedent -> Consequent)

Instruksi Utama:
1. Jawablah setiap pertanyaan pengguna secara akurat, jelas, terstruktur, mendalam, dan edukatif berdasarkan konsep materi perkuliahan dan konteks dokumen yang disediakan.
2. Cantumkan dokumen sumber materi (misal: 2.0 SWTech.pdf, 2.1 AP-RDF.pdf, 6. SPARQL.pdf, dll.) beserta nomor halaman sebagai referensi pada penjelasan Anda.
3. Gunakan format Markdown yang indah dan mudah dibaca:
   - Gunakan **teks tebal** untuk istilah kunci (misal: **Subject**, **Predicate**, **Object**, **TBox**, **ABox**).
   - Gunakan bullet points atau penomoran untuk menjelaskan tahapan/komponen.
   - Gunakan blok kode (```sparql, ```turtle, dll.) untuk sintaks query atau triple RDF.
4. Pertahankan konteks percakapan sebelumnya untuk menjawab pertanyaan lanjutan secara konsisten.
5. Aturan Anti-Halusinasi (Strict Guardrail):
   Jika pengguna menanyakan topik yang SAMA SEKALI DI LUAR materi perkuliahan Semantic Web (contoh: resep masakan, hiburan, gosip, otomotif, atau pertanyaan umum non-akademik), Anda WAJIB menjawab:
   "Maaf, informasi mengenai hal tersebut tidak ditemukan dalam materi perkuliahan yang tersedia."
   JANGAN MENGARANG JAWABAN untuk topik di luar materi kuliah.
6. Analisis Gambar & Multimodal:
   Jika pengguna melampirkan gambar (seperti diagram graf RDF, antarmuka Protege, hierarki ontologi, rumus logika inferensi, atau potongan query SPARQL), analisis visual tersebut secara mendalam, sebutkan komponen yang terlihat, dan kaitkan pembahasannya dengan materi modul perkuliahan yang relevan.
7. Rekomendasi Pertanyaan Lanjutan (Interactive Follow-up Suggestions):
   Di akhir setiap penjelasan materi yang sukses, SELALU berikan 2 hingga 3 rekomendasi pertanyaan selanjutnya yang menarik dan relevan untuk diajukan pengguna, dengan format tepat seperti ini:

   ---
   **Rekomendasi Pertanyaan Lanjutan:**
   * ↳ [Pertanyaan lanjutan 1 yang spesifik dan relevan]
   * ↳ [Pertanyaan lanjutan 2 yang spesifik dan relevan]
   * ↳ [Pertanyaan lanjutan 3 yang spesifik dan relevan]

   (Gunakan simbol panah ↳ persis seperti di atas agar otomatis menjadi tombol klik interaktif di antarmuka web. Jangan cantumkan rekomendasi jika pertanyaan ditolak karena di luar materi kuliah)."""

app = FastAPI(title="KBR Bot - GPT Interface", version="1.0.0")

# [S12] Security Headers Middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [S6] Simple In-Memory Rate Limiter (per user_id)
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

def check_rate_limit(user_id: str) -> bool:
    """Returns True if the request should be rate-limited (denied)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    # Cleanup old entries
    _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if t > window_start]
    if len(_rate_limit_store[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    _rate_limit_store[user_id].append(now)
    return False

# [S7] Input Validation Constants
MAX_MESSAGE_LENGTH = 5000
MAX_IMAGE_DATA_LENGTH = 7 * 1024 * 1024  # ~5MB base64 ≈ 7MB text

@app.middleware("http")
async def user_isolation_middleware(request: Request, call_next):
    """Middleware untuk memastikan setiap client/perangkat memiliki user_id terisolasi."""
    uid = request.headers.get("x-user-id") or request.cookies.get("sw_user_id")
    need_set_cookie = False
    if not uid or len(uid.strip()) < 3:
        uid = "usr_" + uuid.uuid4().hex[:12]
        need_set_cookie = True
    request.state.user_id = uid.strip()
    response = await call_next(request)
    if need_set_cookie or not request.cookies.get("sw_user_id"):
        response.set_cookie(
            key="sw_user_id",
            value=uid.strip(),
            max_age=60 * 60 * 24 * 365,
            path="/",
            httponly=True,
            samesite="lax"
        )
    return response


class ChatRequest(BaseModel):
    message: Optional[str] = ""
    image_data: Optional[str] = None
    image_mime: Optional[str] = "image/jpeg"
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.2


class ChatResponse(BaseModel):
    reply: str
    sources: List[Dict[str, Any]]
    session_id: str
    session_title: str
    search_query_used: Optional[str] = None
    retrieval_method: Optional[str] = "advanced_rag_hybrid_rerank"


PRONOUN_TRIGGERS = {
    "ini", "itu", "tersebut", "dia", "nya", "tadi", "fungsinya", "caranya", 
    "contohnya", "bedanya", "perbedaannya", "mengapa", "kenapa", "bagaimana",
    "kelebihannya", "kekurangannya", "sintaksnya", "maksudnya", "maksud"
}


def rewrite_conversational_query(raw_query: str, session_id: str) -> str:
    """Jika query percakapan merujuk konteks giliran sebelumnya, tulis ulang menjadi query mandiri."""
    if not session_id or not GEMINI_API_KEY:
        return raw_query

    history = get_session_history(session_id, limit=4)
    if not history:
        return raw_query

    # Deteksi apakah query memerlukan penulisan ulang kontekstual
    words = set(re.findall(r"\w+", raw_query.lower()))
    is_referential = bool(words & PRONOUN_TRIGGERS) or len(raw_query.split()) <= 4 or raw_query.lower().startswith((
        "lalu", "kemudian", "selain itu", "kalau", "bagaimana jika", "apakah ada", "apa lagi", "mengapa"
    ))

    if not is_referential:
        return raw_query

    # Buat riwayat singkat 2 turn terakhir
    conv_history_str = ""
    for h in history[-3:]:
        role = "User" if h["role"] == "user" else "Assistant"
        conv_history_str += f"{role}: {h['text'][:140]}\n"

    prompt = f"""Tugas Anda: Tulis ulang pertanyaan lanjutan pengguna menjadi satu kalimat query pencarian mandiri (standalone search query) yang padat kata kunci dalam bahasa Indonesia untuk mencari materi Semantic Web / Knowledge Engineering.
Sertakan entitas / topik utama dari riwayat percakapan sebelumnya.
JANGAN MENJAWAB PERTANYAAN. Keluarkan HANYA 1 baris query pencarian tanpa tanda kutip.

Riwayat Obrolan:
{conv_history_str}

Pertanyaan Lanjutan Pengguna:
{raw_query}

Query Pencarian Mandiri:"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 60
            }
        }
        res = requests.post(url, json=payload, timeout=3.5)
        if res.status_code == 200:
            data = res.json()
            rewritten = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            rewritten = rewritten.strip('"\'`').replace("\n", " ").strip()
            if rewritten and len(rewritten) > 3:
                logger.info(f"[Advanced RAG] Query Rewritten: '{raw_query}' -> '{rewritten}'")
                return rewritten
    except Exception as e:
        logger.warning(f"Gagal rewrite query, fallback ke raw_query: {e}")

    return raw_query


def tokenize_text(text: str) -> List[str]:
    """Tokenisasi kata sederhana untuk BM25."""
    return re.findall(r"\w+", text.lower())


def search_hybrid_rrf(query: str, candidate_limit: int = 40) -> List[Dict[str, Any]]:
    """Mengambil kandidat dengan Dense Vector Search + BM25 Keyword Scoring dan Reciprocal Rank Fusion."""
    try:
        query_vector = list(embedder.embed([query]))[0].tolist()
        if hasattr(qdrant_client, "query_points"):
            response = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=candidate_limit,
                with_payload=True
            )
            points = response.points
        else:
            points, _ = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=candidate_limit,
                with_payload=True
            )

        raw_candidates = []
        seen = set()

        for p in points:
            payload = getattr(p, "payload", {}) or {}
            content = str(payload.get("content", "")).strip()
            source = str(payload.get("source") or "Materi Kuliah")
            page = str(payload.get("page") or "?")
            dense_score = float(getattr(p, "score", 0.0) or 0.0)

            if not content or len(content) < 30 or source.lower() == "blob":
                continue

            key = (source, page, content[:60])
            if key in seen:
                continue
            seen.add(key)

            raw_candidates.append({
                "source": source,
                "page": page,
                "content": content,
                "dense_score": dense_score
            })

        if not raw_candidates:
            return []

        # Hitung Peringkat BM25 pada kandidat
        tokenized_corpus = [tokenize_text(c["content"]) for c in raw_candidates]
        tokenized_query = tokenize_text(query)

        if tokenized_corpus and tokenized_query:
            bm25 = BM25Plus(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_indexed = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_indexed)}
        else:
            bm25_scores = [0.0] * len(raw_candidates)
            bm25_rank_map = {i: i for i in range(len(raw_candidates))}

        # Reciprocal Rank Fusion (RRF)
        # RRF_score = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
        K_RRF = 60
        for dense_rank, cand in enumerate(raw_candidates):
            bm25_rank = bm25_rank_map.get(dense_rank, len(raw_candidates))
            bm25_val = float(bm25_scores[dense_rank]) if dense_rank < len(bm25_scores) else 0.0
            rrf_score = (1.0 / (K_RRF + dense_rank)) + (1.0 / (K_RRF + bm25_rank))
            cand["bm25_score"] = bm25_val
            cand["rrf_score"] = rrf_score

        # Urutkan berdasarkan skor RRF tertinggi
        fused_candidates = sorted(raw_candidates, key=lambda x: x["rrf_score"], reverse=True)
        return fused_candidates
    except Exception as e:
        logger.error(f"Error saat hybrid search Qdrant: {e}")
        return []


def rerank_contexts(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Cross-Encoder Reranking menggunakan FlashRank untuk akurasi tertinggi."""
    if not candidates:
        return []

    # Ambil top 15 kandidat RRF untuk di-rerank
    pool = candidates[:15]

    try:
        passages = [
            {"id": i, "text": c["content"], "meta": c}
            for i, c in enumerate(pool)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = ranker.rerank(rerank_req)

        final_contexts = []
        for res in ranked_results[:top_k]:
            item = res["meta"]
            item["score"] = float(item["dense_score"])
            item["rerank_score"] = float(res["score"])
            final_contexts.append(item)

        return final_contexts
    except Exception as e:
        logger.warning(f"Error saat Cross-Encoder reranking, fallback ke RRF candidates: {e}")
        for c in pool[:top_k]:
            c["score"] = float(c["dense_score"])
            c["rerank_score"] = float(c.get("rrf_score", 0.0))
        return pool[:top_k]


def search_context(raw_query: str, session_id: Optional[str] = None, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    """Pipeline Advanced RAG lengkap: Rewriting -> Hybrid RRF -> FlashRank Reranker."""
    # 1. Conversational Query Rewriting
    effective_query = raw_query
    if session_id:
        effective_query = rewrite_conversational_query(raw_query, session_id)

    # 2. Hybrid Dense + BM25 Search dengan Reciprocal Rank Fusion
    fused_candidates = search_hybrid_rrf(effective_query, candidate_limit=40)

    # 3. Cross-Encoder Reranking (FlashRank)
    final_contexts = rerank_contexts(effective_query, fused_candidates, top_k=top_k)

    return final_contexts, effective_query


def generate_gemini_reply(
    query: str,
    contexts: List[Dict[str, str]],
    session_id: str,
    temperature: float = 0.2,
    image_data: Optional[str] = None,
    image_mime: Optional[str] = "image/jpeg"
) -> str:
    """Kirim prompt dan histori percakapan ke Google Gemini dengan dukungan Multimodal Image dan fallback otomatis."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY belum dikonfigurasi di .env!")

    history = get_session_history(session_id, limit=8)

    # Susun teks konteks RAG
    if contexts:
        context_str = "\n\n".join([
            f"--- [Dokumen Sumber: {c['source']} | Halaman: {c['page']}] ---\n{c['content']}"
            for c in contexts
        ])
    else:
        context_str = "Tidak ada dokumen yang ditemukan."

    user_augmented_prompt = f"""Konteks Dokumen Perkuliahan:
{context_str}

Pertanyaan Pengguna:
{query}"""

    # Buat contents array untuk multi-turn Gemini API
    # Ambil maksimal 8 percakapan terakhir untuk sliding window memory
    recent_history = history[-8:]
    contents = []

    for turn in recent_history:
        role = "user" if turn["role"] == "user" else "model"
        turn_text = turn.get("text", "")
        if turn.get("has_image"):
            turn_text = f"[Pengguna melampirkan gambar pada giliran ini]\n{turn_text}"
        contents.append({
            "role": role,
            "parts": [{"text": turn_text}]
        })

    # Siapkan parts pesan pengguna saat ini (termasuk Gambar jika ada)
    user_parts = []
    if image_data:
        clean_b64 = image_data
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",", 1)[1]
        user_parts.append({
            "inlineData": {
                "mimeType": image_mime or "image/jpeg",
                "data": clean_b64.strip()
            }
        })

    user_parts.append({"text": user_augmented_prompt})

    # Tambahkan pesan sekarang yang sudah diaugmentasi konteks RAG dan gambar
    contents.append({
        "role": "user",
        "parts": user_parts
    })

    # Map alias model lama yang sudah di-deprecate Google ke model aktif
    MODEL_FALLBACK_MAP = {
        "gemini-2.5-flash-lite": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        "gemini-2.5-pro": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"],
        "gemini-2.0-flash": ["gemini-flash-latest", "gemini-flash-lite-latest"],
        "gemini-2.0-flash-lite": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"],
        "gemini-1.5-flash": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"],
        "gemini-1.5-pro": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"],
    }

    # Urutan prioritas model Flash Lite dan Flash yang aktif di Google Gemini API
    base_models = [
        GEMINI_MODEL,
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ]

    seen_models = set()
    candidate_models = []
    for m in base_models:
        clean_m = m.replace("models/", "")
        if clean_m not in seen_models:
            seen_models.add(clean_m)
            candidate_models.append(clean_m)
        for alias in MODEL_FALLBACK_MAP.get(clean_m, []):
            if alias not in seen_models:
                seen_models.add(alias)
                candidate_models.append(alias)

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        }
    }

    last_err = ""
    for model_name in candidate_models:
        # [S3] API key dikirim via header, bukan di URL query parameter
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                else:
                    err_desc = resp.text[:120]
                    try:
                        err_json = resp.json().get("error", {})
                        err_desc = err_json.get("message", err_desc)
                    except Exception:
                        pass
                    last_err = f"{model_name} HTTP {resp.status_code}: {err_desc}"

                    if resp.status_code == 404:
                        # Model sudah tidak tersedia atau deprecate untuk user baru, jangan retry, langsung coba model berikutnya
                        break
                    elif resp.status_code in (429, 503):
                        time.sleep(1.2)
                        continue
            except Exception as e:
                last_err = f"{model_name} exception: {str(e)}"
                time.sleep(1)

        logger.warning(f"Model {model_name} tidak tersedia ({last_err}), beralih ke model berikutnya...")

    raise RuntimeError(f"Semua model gagal merespons: {last_err}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    """Endpoint chat utama yang dipanggil oleh antarmuka web ChatGPT (Mendukung Teks + Gambar Multimodal)."""
    user_id = get_user_id(request)

    # [S6] Rate limiting check
    if check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Terlalu banyak permintaan. Silakan tunggu sebentar.")

    query = (req.message or "").strip()
    image_data = req.image_data
    image_mime = req.image_mime or "image/jpeg"

    # [S7] Input length validation
    if query and len(query) > MAX_MESSAGE_LENGTH:
        query = query[:MAX_MESSAGE_LENGTH]
    if image_data and len(image_data) > MAX_IMAGE_DATA_LENGTH:
        raise HTTPException(status_code=413, detail="Ukuran gambar terlalu besar (maks ~5MB).")

    if not query and not image_data:
        raise HTTPException(status_code=400, detail="Pesan teks atau gambar tidak boleh kosong.")

    # Jika user hanya mengirim gambar tanpa teks pertanyaan, berikan prompt instruksi default
    if not query and image_data:
        query = "Jelaskan dan analisis gambar materi perkuliahan ini secara mendalam berdasarkan konsep Semantic Web & Knowledge Engineering."

    # Manajemen Session ID
    session_id = req.session_id or str(uuid.uuid4())
    title_prefix = "[Gambar] " if image_data else ""
    session_title = title_prefix + (query[:30] + ("..." if len(query) > 30 else "")) or "Percakapan Baru"

    # 1. Pipeline Advanced RAG (Conversational Rewriter + Hybrid BM25/Dense RRF + FlashRank Reranker)
    contexts, search_query_used = search_context(raw_query=query, session_id=session_id, top_k=4)

    # 2. Pemanggilan LLM Gemini (Vision Multimodal + RAG)
    try:
        reply = generate_gemini_reply(
            query=query,
            contexts=contexts,
            session_id=session_id,
            temperature=req.temperature or 0.2,
            image_data=image_data,
            image_mime=image_mime
        )
    except Exception as e:
        logger.error(f"Error saat inferensi Gemini: {e}")
        reply = f"Terjadi kesalahan saat memproses permintaan: {str(e)}"

    # 3. Simpan ke Database SQLite dengan isolasi user_id
    save_session_turn(
        session_id=session_id,
        user_id=user_id,
        title=session_title,
        user_msg=query,
        has_image=bool(image_data),
        bot_reply=reply
    )

    # Siapkan data sumber sitasi yang bersih dengan skor reranker
    clean_sources = []
    seen_sources = set()
    for c in contexts:
        key = (c["source"], c["page"])
        if key not in seen_sources:
            seen_sources.add(key)
            snippet = c["content"][:200].replace("\n", " ")
            clean_sources.append({
                "source": c["source"],
                "page": c["page"],
                "snippet": snippet + "...",
                "rerank_score": round(float(c.get("rerank_score", 0.0)), 4)
            })

    return ChatResponse(
        reply=reply,
        sources=clean_sources,
        session_id=session_id,
        session_title=session_title,
        search_query_used=search_query_used,
        retrieval_method="advanced_rag_hybrid_rerank"
    )


@app.get("/api/sessions")
async def list_sessions(request: Request):
    """Mengembalikan daftar riwayat sesi obrolan yang HANYA dimiliki oleh user ini."""
    user_id = get_user_id(request)
    sessions = get_user_sessions_list(user_id)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Mengambil riwayat pesan dalam satu sesi milik user ini."""
    user_id = get_user_id(request)
    detail = get_user_session_detail(session_id, user_id)
    if not detail:
        return {"session_id": session_id, "title": "Percakapan Baru", "messages": []}
    return detail


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Menghapus sesi tertentu milik user ini."""
    user_id = get_user_id(request)
    delete_user_session_db(session_id, user_id)
    return {"status": "ok", "deleted": session_id}


@app.post("/api/clear")
async def clear_all_sessions(request: Request):
    """Mengosongkan seluruh riwayat obrolan milik user ini saja."""
    user_id = get_user_id(request)
    clear_all_user_sessions_db(user_id)
    return {"status": "ok", "message": "Semua sesi Anda dibersihkan."}


@app.get("/api/health")
async def health_check():
    """Memeriksa koneksi Qdrant dan Gemini."""
    try:
        col_info = qdrant_client.get_collection(QDRANT_COLLECTION)
        qdrant_ok = True
        vector_count = col_info.points_count
    except Exception as e:
        qdrant_ok = False
        vector_count = 0

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "qdrant_connected": qdrant_ok,
        "collection": QDRANT_COLLECTION,
        "points_count": vector_count,
        "gemini_model": GEMINI_MODEL,
        "gemini_api_key_set": bool(GEMINI_API_KEY)
    }


# Mount antarmuka statis HTML/CSS/JS di root /
STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Memulai Server RAG Web GPT di http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
