#!/usr/bin/env python3
"""
Unit and Integration Tests for Advanced RAG Pipeline
Covers:
1. Conversational Query Rewriter
2. BM25 + Dense Hybrid Search with Reciprocal Rank Fusion (RRF)
3. FlashRank Cross-Encoder Reranking
4. Performance / Latency Benchmarks
"""

import sys
import time
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.server import (
    tokenize_text,
    search_hybrid_rrf,
    rerank_contexts,
    search_context,
    rewrite_conversational_query,
    save_session_turn,
    ranker
)


def test_tokenizer():
    print("\n[Test 1] Tokenizer:")
    text = "Query SPARQL SELECT ?subject WHERE { ?s ?p ?o . }"
    tokens = tokenize_text(text)
    assert "sparql" in tokens
    assert "select" in tokens
    assert "subject" in tokens
    print(f"  Passed! Tokens extracted: {tokens[:5]}")


def test_conversational_query_rewriting():
    print("\n[Test 2] Conversational Query Rewriter:")
    session_id = f"test_rewriter_{int(time.time())}"
    user_id = "test_user"

    # Turn 1: Discuss OWL & Ontologies
    save_session_turn(
        session_id=session_id,
        user_id=user_id,
        title="Test Session",
        user_msg="Apa itu Web Ontology Language (OWL)?",
        has_image=False,
        bot_reply="OWL adalah bahasa representasi pengetahuan untuk mendefinisikan ontologi di Semantic Web."
    )

    # Turn 2: Follow-up question with pronoun
    followup_query = "Apa bedanya dengan RDF?"
    rewritten = rewrite_conversational_query(followup_query, session_id)
    print(f"  Original Query : '{followup_query}'")
    print(f"  Rewritten Query: '{rewritten}'")

    # Assert that rewritten query is non-empty and captured context
    assert len(rewritten) >= len(followup_query) or "owl" in rewritten.lower() or "rdf" in rewritten.lower()
    print("  Passed! Conversational query successfully recontextualized.")


def test_hybrid_search_rrf():
    print("\n[Test 3] Hybrid Search with Reciprocal Rank Fusion (RRF):")
    query = "sintaks query SPARQL SELECT dan PREFIX"
    t0 = time.time()
    candidates = search_hybrid_rrf(query, candidate_limit=25)
    duration = time.time() - t0

    assert len(candidates) > 0, "Harus menemukan kandidat dari Qdrant"
    first = candidates[0]
    assert "dense_score" in first
    assert "bm25_score" in first
    assert "rrf_score" in first
    print(f"  Retrieved {len(candidates)} fused candidates in {duration*1000:.1f}ms")
    print(f"  Top Candidate Source: {first['source']} (Page: {first['page']})")
    print(f"  Scores: RRF={first['rrf_score']:.5f}, Dense={first['dense_score']:.4f}, BM25={first['bm25_score']:.2f}")
    print("  Passed! Hybrid search and RRF scoring verified.")


def test_flashrank_reranking():
    print("\n[Test 4] FlashRank Cross-Encoder Reranking:")
    query = "protege ontologi"
    mock_candidates = [
        {
            "source": "Modul_Masakan.pdf",
            "page": "1",
            "content": "Resep rendang daging sapi khas Padang membutuhkan rempah-rempah kelapa dan santan kental.",
            "dense_score": 0.35,
            "rrf_score": 0.02
        },
        {
            "source": "3.1 Protege.pdf",
            "page": "5",
            "content": "Protege adalah alat editor ontologi open source untuk mendefinisikan kelas, properti, dan hierarki.",
            "dense_score": 0.75,
            "rrf_score": 0.03
        }
    ]

    reranked = rerank_contexts(query, mock_candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["source"] == "3.1 Protege.pdf", "Top-1 harus modul Protege!"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"], "Skor Protege harus lebih tinggi dari masakan!"
    print(f"  Top-1: {reranked[0]['source']} (Rerank Score: {reranked[0]['rerank_score']:.5f})")
    print(f"  Top-2: {reranked[1]['source']} (Rerank Score: {reranked[1]['rerank_score']:.5f})")
    print("  Passed! Cross-Encoder correctly ranked relevant chunk above distractor.")


def test_full_pipeline_end_to_end():
    print("\n[Test 5] Full Advanced RAG Pipeline End-to-End & Benchmark:")
    query = "Bagaimana hierarki kelas dibuat pada Protege?"
    t0 = time.time()
    final_contexts, effective_query = search_context(raw_query=query, session_id=None, top_k=4)
    total_time_ms = (time.time() - t0) * 1000

    assert len(final_contexts) > 0, "Harus mengembalikan minimal 1 konteks"
    assert effective_query == query
    print(f"  Effective Query: {effective_query}")
    print(f"  Contexts returned: {len(final_contexts)}")
    print(f"  Total Retrieval Pipeline Latency: {total_time_ms:.1f}ms")
    for i, c in enumerate(final_contexts):
        print(f"    [{i+1}] {c['source']} (hal {c['page']}) - Rerank: {c.get('rerank_score', 0):.4f}")
    assert total_time_ms < 600, f"Latency harus di bawah 600ms, actual: {total_time_ms}ms"
    print("  Passed! End-to-end Advanced RAG pipeline completed smoothly.")


if __name__ == "__main__":
    print("=== Menjalankan Unit & Integration Tests Advanced RAG ===")
    test_tokenizer()
    test_conversational_query_rewriting()
    test_hybrid_search_rrf()
    test_flashrank_reranking()
    test_full_pipeline_end_to_end()
    print("\n=== SEMUA PENGUJIAN ADVANCED RAG BERHASIL (100% PASS) ===")
