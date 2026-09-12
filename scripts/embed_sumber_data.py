#!/usr/bin/env python3
"""
Batch Ingestion Script for Sumber_Data
Sistem RAG Bot Telegram Berbasis n8n
Sesuai PRD Section 6 (FR-1, FR-2):
- Parsing 12 dokumen PDF materi Semantic Web
- Recursive text chunking (chunk_size: 800-1000, overlap: 100)
- Vector embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Upload ke Vector Database Qdrant (local atau Qdrant Cloud)
"""

import os
import sys
import glob
import json
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "Sumber_Data"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 100


def extract_text_from_pdf(pdf_path: str):
    """Ekstraksi teks dari PDF per halaman dengan fallback otomatis."""
    import pypdf
    pages_data = []

    try:
        reader = pypdf.PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_data.append({
                    "page": idx + 1,
                    "text": text
                })
        if pages_data:
            return pages_data
    except Exception as e:
        # Fallback jika PDF truncated (misal 5. Knowledge Base.pdf yang missing startxref)
        pass

    # Stream CMap Recovery Fallback
    try:
        import zlib, re
        with open(pdf_path, 'rb') as f:
            data = f.read()

        cmaps = {}
        for m in re.finditer(rb'(\d+)\s+0\s+obj\s*<<.*?>>\s*stream[\r\n]+(.*?)[\r\n]+endstream', data, re.DOTALL):
            try:
                dec = zlib.decompress(m.group(2))
                if b'begincmap' in dec:
                    for line in dec.decode('latin1', errors='ignore').splitlines():
                        bf = re.match(r'<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>', line.strip())
                        if bf:
                            src = bf.group(1).upper()
                            dst_hex = bf.group(2)
                            chars = ''.join(chr(int(dst_hex[i:i+4], 16)) for i in range(0, len(dst_hex), 4))
                            cmaps[src] = chars
            except Exception:
                pass

        content_pattern = rb'obj\s*<<[^>]*?/Filter\s*/FlateDecode[^>]*?>>\s*stream[\r\n]+(.*?)[\r\n]+endstream'
        page_idx = 1
        for m in re.finditer(content_pattern, data, re.DOTALL):
            try:
                dec = zlib.decompress(m.group(1)).decode('latin1', errors='ignore')
                if 'BT' in dec and 'ET' in dec:
                    text_ops = re.findall(r'(\[.*?\]\s*TJ|<[0-9A-Fa-f]+>\s*Tj)', dec)
                    words = []
                    for op in text_ops:
                        hexes = re.findall(r'<([0-9A-Fa-f]+)>', op)
                        phrase = ''
                        for h in hexes:
                            for i in range(0, len(h), 2):
                                phrase += cmaps.get(h[i:i+2].upper(), '')
                        if phrase.strip():
                            words.append(phrase.strip())
                    clean_page = ' '.join(words).strip()
                    if len(clean_page) > 20:
                        pages_data.append({"page": page_idx, "text": clean_page})
                        page_idx += 1
            except Exception:
                pass
    except Exception:
        pass

    return pages_data


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Recursive-like chunking dengan pemisah paragraf/kalimat."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Cari breakpoint alami (newline, titik, atau spasi)
        breakpoint = -1
        for delim in ["\n\n", "\n", ". ", " "]:
            pos = text.rfind(delim, start + chunk_size // 2, end)
            if pos != -1:
                breakpoint = pos + len(delim)
                break

        if breakpoint == -1 or breakpoint <= start:
            breakpoint = end

        chunk = text[start:breakpoint].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, breakpoint - overlap)

    return chunks


class EmbeddingProvider:
    """Provider embedding: mencoba FastEmbed lokal (cepat & offline), jika gagal pakai HF API."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_token: str = ""):
        self.model_name = model_name
        self.hf_token = hf_token
        self.fastembed_model = None
        self.vector_size = 384

        try:
            from fastembed import TextEmbedding
            print(f"[*] Menginisialisasi FastEmbed lokal ({model_name})...")
            self.fastembed_model = TextEmbedding(model_name=self.model_name)
            print("[+] FastEmbed lokal siap digunakan.")
        except Exception as e:
            print(f"[!] FastEmbed lokal tidak tersedia ({e}). Menggunakan HuggingFace API...")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.fastembed_model is not None:
            # Gunakan fastembed lokal
            embeddings = list(self.fastembed_model.embed(texts))
            return [emb.tolist() for emb in embeddings]
        else:
            # Fallback ke HuggingFace Inference API
            import requests
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            
            # Batch per 16 chunk untuk menjaga batas HF
            all_embeddings = []
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                res = requests.post(api_url, headers=headers, json={"inputs": batch, "options": {"wait_for_model": True}})
                if res.status_code == 200:
                    data = res.json()
                    all_embeddings.extend(data)
                else:
                    raise RuntimeError(f"Gagal generate embedding via HuggingFace API: {res.status_code} - {res.text}")
                time.sleep(0.5)
            return all_embeddings


def main():
    print("=" * 60)
    print("   BATCH INGESTION: SUMBER_DATA -> QDRANT VECTOR STORE")
    print("=" * 60)

    data_dir_path = Path(DATA_DIR)
    if not data_dir_path.exists():
        print(f"[!] Folder Sumber_Data tidak ditemukan di: {data_dir_path.resolve()}")
        sys.exit(1)

    pdf_files = sorted(list(data_dir_path.glob("*.pdf")))
    print(f"[*] Ditemukan {len(pdf_files)} dokumen PDF di {data_dir_path.resolve()}:")
    for f in pdf_files:
        print(f"   - {f.name} ({f.stat().st_size // 1024} KB)")

    if not pdf_files:
        print("[!] Tidak ada file PDF untuk diproses.")
        sys.exit(0)

    # 1. Ekstraksi dan Chunking
    all_chunks = []
    print("\n[*] Langkah 1: Ekstraksi teks & Chunking...")
    for pdf_path in pdf_files:
        pages = extract_text_from_pdf(str(pdf_path))
        doc_chunks_count = 0
        for page_info in pages:
            chunks = chunk_text(page_info["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for c_idx, c_text in enumerate(chunks):
                # ID Deterministik (UUID v5) agar jika di-embed ulang, Qdrant melakukan overwrite/upsert dan tidak menduplikat data
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.name}_p{page_info['page']}_c{c_idx}"))
                all_chunks.append({
                    "id": chunk_id,
                    "text": c_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_info["page"],
                        "chunk_index": c_idx,
                        "content": c_text
                    }
                })
                doc_chunks_count += 1
        print(f"   [OK] {pdf_path.name}: {len(pages)} halaman -> {doc_chunks_count} chunk")

    print(f"\n[+] Total chunk unik yang dihasilkan: {len(all_chunks)} chunk")

    # 2. Inisialisasi Vector DB Qdrant
    print(f"\n[*] Langkah 2: Menghubungkan ke Qdrant ({QDRANT_HOST})...")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("[!] Modul qdrant-client belum terpasang. Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    try:
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=30)
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]

        if QDRANT_COLLECTION in existing_names:
            print(f"[*] Membersihkan collection lama '{QDRANT_COLLECTION}' agar bebas duplikasi...")
            client.delete_collection(collection_name=QDRANT_COLLECTION)

        print(f"[*] Membuat collection baru yang bersih: '{QDRANT_COLLECTION}' (Dimensi: 384, Jarak: Cosine)...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[+] Collection '{QDRANT_COLLECTION}' siap digunakan.")
    except Exception as e:
        print(f"[!] Gagal menghubungkan ke Qdrant: {e}")
        print("[TIP] Pastikan kontainer docker 'qdrant' sudah berjalan: docker compose up -d qdrant")
        sys.exit(1)

    # 3. Generate Embeddings & Upsert ke Qdrant
    print(f"\n[*] Langkah 3: Generate Embedding ({EMBEDDING_MODEL_NAME}) & Upsert...")
    embedder = EmbeddingProvider(EMBEDDING_MODEL_NAME, HUGGINGFACE_API_KEY)

    batch_size = 32
    total_upserted = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [item["text"] for item in batch]
        
        vectors = embedder.embed_texts(texts)
        
        points = [
            PointStruct(
                id=item["id"],
                vector=vectors[idx],
                payload=item["metadata"]
            )
            for idx, item in enumerate(batch)
        ]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        total_upserted += len(points)
        print(f"   -> Terunggah {total_upserted}/{len(all_chunks)} chunk...")

    print(f"\n[+] Sukses! Seluruh {total_upserted} chunk berhasil disimpan di Qdrant.")

    # 4. Uji Verifikasi Similarity Search
    print("\n[*] Langkah 4: Uji Verifikasi Retrieval (RAG Query Test)...")
    test_query = "Apa itu Resource Description Framework (RDF) dan ontologi?"
    query_vector = embedder.embed_texts([test_query])[0]
    
    try:
        if hasattr(client, "query_points"):
            search_response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=2
            )
            search_results = search_response.points
        else:
            search_results = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=2
            )

        print(f"   Query Uji: \"{test_query}\"")
        print(f"   Hasil Retrieval Teratas:")
        for idx, res in enumerate(search_results):
            score = getattr(res, "score", 0.0)
            payload = getattr(res, "payload", {}) or {}
            source = payload.get('source', 'unknown')
            page = payload.get('page', '?')
            print(f"   [{idx + 1}] Skor: {score:.4f} | Sumber: {source} (Hal. {page})")
            snippet = payload.get('content', '')[:160].replace('\n', ' ')
            safe_snippet = snippet.encode('ascii', errors='replace').decode('ascii')
            print(f"       Cuplikan: {safe_snippet}...")
    except Exception as e:
        print(f"   [!] Info uji retrieval: {e}")

    print("\n" + "=" * 60)
    print("   INGESTION SELESAI & KNOWLEDGE BASE SIAP DIGUNAKAN!")
    print("=" * 60)


if __name__ == "__main__":
    main()
