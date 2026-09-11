# Product Requirements Document (PRD)
## Sistem RAG Bot Telegram Berbasis n8n

**Pemilik Proyek:** Rex (Theodosius Rexy Mahardika)
**Versi:** 1.1
**Tanggal:** 10 September 2026

---

## 1. Latar Belakang

Dibutuhkan sistem chatbot berbasis Retrieval-Augmented Generation (RAG) yang dapat menjawab pertanyaan banyak pengguna secara bersamaan berdasarkan satu dokumen/knowledge base yang sama, dijalankan sepenuhnya dengan komponen gratis, dan diakses melalui Telegram.

## 2. Tujuan

- Membangun bot Telegram yang dapat menjawab pertanyaan berbasis konten dokumen yang di-upload.
- Mendukung banyak user bertanya secara bersamaan ke 1 knowledge base yang sama, tanpa histori percakapan antar user saling tercampur.
- Meminimalkan biaya operasional (target: Rp0 untuk software, hanya biaya opsional untuk domain).
- Sistem dapat mengingat konteks percakapan dalam satu sesi chat, terisolasi per user/grup.
- Jawaban bot berbasis dokumen sumber, bukan halusinasi model.

## 3. Ruang Lingkup

### Termasuk (In Scope)
- Workflow ingestion dokumen (PDF/TXT → chunking → embedding → vector store), dengan kontrol akses hanya admin.
- Workflow chat retrieval + generation via Telegram, mendukung multi-user dengan 1 knowledge base bersama.
- Isolasi memory percakapan per chat_id (window buffer per sesi).
- Deployment self-hosted di VPS dengan webhook publik, mode Queue (Postgres + Redis) untuk menangani eksekusi paralel banyak user.

### Tidak Termasuk (Out of Scope)
- Integrasi WhatsApp (diputuskan untuk fase ini hanya Telegram).
- Dokumen/knowledge base privat per user (semua user akses knowledge base yang sama).
- UI dashboard admin kustom (pengelolaan tetap via n8n editor).

## 4. Arsitektur Sistem

**Workflow 1 — Ingestion (Persiapan Dokumen)**
```
Upload PDF/TXT → Text Splitter (chunking) → HF Embeddings → Qdrant/Supabase Insert
```

**Workflow 2 — Retrieval & Generation (Chat Bot)**
```
Telegram Trigger → AI Agent Node
  ├─ Model: OpenRouter (:free, via OpenAI Chat Model node)
  ├─ Memory: Window Buffer Memory (5–10 pesan terakhir)
  │   Session Key = {{ $json.message.chat.id }}  → isolasi memori per user/grup
  └─ Tool: Vector Store Tool (Qdrant + HF Embeddings, 1 collection bersama)
→ Telegram Send Message
```

**Deployment — Queue Mode (untuk eksekusi paralel banyak user)**
```
Telegram webhook → cloudflared → n8n-main (terima webhook, dorong job ke queue)
                                      ↓
                                   Redis (antrian job)
                                      ↓
                              n8n-worker (eksekusi workflow, bisa di-scale >1)
                                      ↓
                                Postgres (DB workflow & execution, wajib untuk queue mode)
```

## 5. Tech Stack

| Komponen | Pilihan | Keterangan |
|---|---|---|
| Workflow engine | n8n (self-hosted, Docker), mode **Queue** | Di VPS Oracle Cloud Always Free |
| Eksekusi workflow | `n8n-main` (terima webhook) + `n8n-worker` (eksekusi, bisa di-scale) | Memisahkan penerimaan request dan eksekusi agar tahan banyak user bersamaan |
| Database internal n8n | Postgres | Wajib untuk queue mode (SQLite default tidak support akses concurrent) |
| Job queue | Redis | Antrian eksekusi antara n8n-main dan n8n-worker |
| Exposure webhook | Cloudflare Tunnel | Gratis, auto-SSL, tanpa buka port |
| Interface chat | Telegram Bot API | Setup via @BotFather |
| Vector database | Qdrant Cloud (free tier) atau Supabase pgvector | 1 collection bersama untuk semua user |
| Embedding model | Hugging Face Inference Providers — `sentence-transformers/all-MiniLM-L6-v2` | Endpoint feature-extraction (masih gratis) |
| LLM generator | OpenRouter API, model `:free` (dicek berkala, tidak di-hardcode) | Fallback ke model kedua jika model utama di-delist |

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | Sistem dapat menerima upload dokumen PDF/TXT dan memecahnya menjadi chunk (~500–1000 karakter, overlap ~100 karakter). |
| FR-2 | Sistem dapat menghasilkan vector embedding untuk setiap chunk dan menyimpannya ke vector database. |
| FR-3 | Bot Telegram dapat menerima pesan teks dari user secara real-time. |
| FR-4 | Sistem melakukan similarity search (top-k 3–5 chunk) untuk mencari konteks relevan sebelum menjawab. |
| FR-5 | Sistem mengirim konteks + histori percakapan + pertanyaan ke LLM via OpenRouter dan mengembalikan jawaban ke user. |
| FR-6 | Bot menjawab "tidak tahu" jika tidak ada konteks relevan ditemukan (mengurangi halusinasi). |
| FR-7 | Sistem menyimpan histori percakapan sementara (5–10 pesan terakhir) per sesi chat. |
| FR-8 | Sistem mendukung banyak user/grup bertanya secara bersamaan tanpa histori percakapan saling tercampur (isolasi via session key = chat_id). |
| FR-9 | Hanya admin (whitelist chat_id) yang dapat memicu proses ingestion dokumen baru ke knowledge base bersama. |

## 7. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Biaya operasional software = Rp0 (semua komponen tier gratis). |
| NFR-2 | Webhook n8n harus dapat diakses publik via HTTPS (Cloudflare Tunnel). |
| NFR-3 | Sistem harus punya mekanisme fallback model jika model `:free` OpenRouter di-delist. |
| NFR-4 | Rate limit API (OpenRouter ~20 req/menit, HF free tier) harus diperhitungkan agar bot tidak error saat traffic naik. |
| NFR-5 | Response time bot maksimal ~10–15 detik per pesan (batas wajar untuk model gratis). |
| NFR-6 | Sistem harus dapat memproses beberapa eksekusi workflow paralel (multi-user) tanpa saling menimpa data memory — dicapai dengan n8n Queue Mode (Postgres + Redis + worker terpisah). |
| NFR-7 | Jumlah worker n8n harus dapat di-scale horizontal (`docker compose up -d --scale n8n-worker=N`) jika volume user bertambah. |

## 8. Milestone / Tahapan Implementasi

1. Provisioning VPS (Oracle Cloud Always Free) + install Docker.
2. Deploy n8n dengan Queue Mode (Postgres + Redis + n8n-main + n8n-worker) via docker-compose, sambungkan Cloudflare Tunnel.
3. Setup akun: BotFather, OpenRouter, Qdrant Cloud, Hugging Face.
4. Bangun & uji Workflow Ingestion dengan 1–2 dokumen kecil, tambahkan IF node whitelist admin chat_id.
5. Bangun & uji Workflow Chat (Telegram Trigger → AI Agent → Reply), set Session Key Window Buffer Memory = `chat.id`.
6. Testing end-to-end dengan dokumen nyata, termasuk uji beberapa user chat bersamaan.
7. Tuning prompt, top-k retrieval, dan window memory.
8. (Opsional, jika volume user naik) Scale jumlah `n8n-worker` dan isi saldo kecil OpenRouter untuk rate limit lebih tinggi.

## 9. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Model `:free` OpenRouter di-delist tanpa notifikasi | Cek daftar model berkala, siapkan model fallback di workflow |
| Rate limit API (OpenRouter/HF) tercapai saat traffic naik | Isi saldo kecil OpenRouter untuk limit lebih tinggi bila perlu |
| Webhook n8n tidak stabil (tunnel putus) | Gunakan Cloudflare Tunnel (lebih stabil dari ngrok free) |
| Storage Qdrant Cloud free tier terbatas (~1GB) | Batasi jumlah/ukuran dokumen atau pindah ke Supabase pgvector bila perlu |
| Jawaban bot halusinasi di luar konteks dokumen | System prompt eksplisit + threshold similarity score |
| User non-admin ikut menambah dokumen ke knowledge base bersama | IF node whitelist chat_id admin sebelum workflow ingestion dijalankan |
| Kuota API (OpenRouter/HF) habis lebih cepat karena dipakai banyak user bersamaan | Monitor penggunaan harian, isi saldo kecil OpenRouter bila perlu |
| Eksekusi paralel menumpuk dan memperlambat respons saat user banyak | Queue Mode sudah mengatasi ini di level arsitektur; tambah jumlah worker jika perlu |

## 10. Metrik Keberhasilan

- Bot berhasil menjawab pertanyaan berbasis dokumen dengan akurasi kontekstual yang relevan (evaluasi manual/qualitative).
- Beberapa user dapat chat bersamaan tanpa histori percakapan saling tercampur (diverifikasi lewat testing paralel).
- Tidak ada downtime signifikan akibat webhook/tunnel dalam periode testing.
- Biaya operasional tetap Rp0 (di luar biaya domain opsional).

## 11. Referensi File Deployment

- `docker-compose.yml` — definisi service Postgres, Redis, n8n-main, n8n-worker, dan cloudflared.
- `.env.example` — template environment variable yang perlu diisi sebelum deployment.
