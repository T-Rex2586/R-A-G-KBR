#!/usr/bin/env python3
"""
Test RAG Pipeline Script
Memverifikasi koneksi Qdrant, status collection knowledge_base,
dan melakukan similarity search pengujian terhadap materi Semantic Web.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")


def test_pipeline():
    print("=" * 60)
    print("   UJI PIPELINE RAG: QDRANT & KNOWLEDGE BASE")
    print("=" * 60)

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("[!] Modul qdrant-client belum terpasang. Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[*] Menghubungkan ke Qdrant di {QDRANT_HOST}...")
    try:
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=10)
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        print(f"[+] Berhasil terhubung. Collections tersedia: {col_names}")

        if QDRANT_COLLECTION not in col_names:
            print(f"[!] Collection '{QDRANT_COLLECTION}' belum dibuat di Qdrant.")
            print("    Jalankan batch embedding terlebih dahulu: python scripts/embed_sumber_data.py")
            return

        info = client.get_collection(QDRANT_COLLECTION)
        print(f"[+] Status Collection '{QDRANT_COLLECTION}':")
        print(f"    - Jumlah Vektor: {info.points_count}")
        print(f"    - Status: {info.status}")

        if info.points_count == 0:
            print("[!] Collection masih kosong. Jalankan: python scripts/embed_sumber_data.py")
            return

        # Contoh query interaktif/uji
        sample_queries = [
            "Apa itu Resource Description Framework (RDF) dan apa saja komponen Triple?",
            "Bagaimana cara melakukan query menggunakan SPARQL?",
            "Apa perbedaan antara Ontologi dan Knowledge Base?",
            "Jelaskan aturan inferensi dalam SWRL (Semantic Web Rule Language)!"
        ]

        try:
            from fastembed import TextEmbedding
            embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            embedder = None

        print("\n[*] Menjalankan uji similarity search pada sampel pertanyaan materi:")
        for q in sample_queries:
            print(f"\n>> Pertanyaan: \"{q}\"")
            if embedder is not None and hasattr(client, "query_points"):
                q_vec = list(embedder.embed([q]))[0].tolist()
                resp = client.query_points(collection_name=QDRANT_COLLECTION, query=q_vec, limit=1)
                matches = resp.points
            else:
                matches, _ = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

            if matches:
                p = getattr(matches[0], "payload", {}) or {}
                source = p.get('source', 'unknown')
                page = p.get('page', '?')
                print(f"   [Top Match] Dokumen: {source} (Hal. {page})")
                snippet = p.get('content', '')[:160].replace('\n', ' ')
                safe_snippet = snippet.encode('ascii', errors='replace').decode('ascii')
                print(f"   Cuplikan: {safe_snippet}...")

        print("\n[+] Pipeline Qdrant siap melayani n8n AI Agent Chat Bot!")

    except Exception as e:
        print(f"[!] Gagal terhubung ke Qdrant: {e}")
        print("[TIP] Pastikan service Qdrant di docker-compose sudah aktif:")
        print("      docker compose up -d qdrant")


if __name__ == "__main__":
    test_pipeline()
