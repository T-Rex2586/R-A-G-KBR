# Sistem RAG Bot Telegram Berbasis n8n (Queue Mode)

Implementasi lengkap sistem Chatbot berbasis **Retrieval-Augmented Generation (RAG)** menggunakan **n8n** (arsitektur **Queue Mode**), **PostgreSQL**, **Redis**, **Qdrant Vector Database**, **Hugging Face Embeddings**, dan **OpenRouter Free Tier LLM**, diakses melalui **Telegram Bot API** dengan tunneling **Cloudflare Tunnel**.

Sesuai spesifikasi teknis pada dokumen [PRD_RAG_Bot_Telegram_n8n.md](file:///c:/nlp/Tugas_RAG/PRD_RAG_Bot_Telegram_n8n.md).

---

## 📑 Daftar Isi
1. [Arsitektur Sistem](#-arsitektur-sistem)
2. [Fitur Utama](#-fitur-utama)
3. [Struktur Repositori](#-struktur-repositori)
4. [Kebutuhan Sistem & Akun API](#-kebutuhan-sistem--akun-api)
5. [Langkah Instalasi & Deployment](#-langkah-instalasi--deployment)
6. [Ingestion & Batch Embedding 12 Dokumen PDF](#-ingestion--batch-embedding-12-dokumen-pdf)
7. [Import & Setup Workflow di n8n](#-import--setup-workflow-di-n8n)
8. [Panduan Pengujian (Multi-User & RAG)](#-panduan-pengujian-multi-user--rag)
9. [Scaling & Manajemen Operasional](#-scaling--manajemen-operasional)

---

## 🏛️ Arsitektur Sistem

Sistem ini didesain dengan pemisahan peran (*separation of concerns*) menggunakan **n8n Queue Mode** untuk menangani eksekusi multi-user secara bersamaan tanpa lag atau bentrok memori:

```
Telegram Users (Multi-User)
       │ (HTTPS Webhook)
       ▼
Cloudflare Tunnel (cloudflared)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ n8n-main (Port 5678)                                   │
│  - Menerima Webhook Telegram                           │
│  - Web UI Editor & Manajemen Workflow                  │
│  - Mendorong job eksekusi ke antrian Redis             │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────┐         ┌────────────────────────┐
│ Redis (Job Queue)     │         │ PostgreSQL             │
│ Antrian job asinkron  │◄────────┤ Database state & run   │
└───────────┬───────────┘         └───────────▲────────────┘
            │                                 │
            ▼                                 │
┌────────────────────────────────────────┐    │
│ n8n-worker (Bisa di-scale > 1)         ├────┘
│  - Mengambil job dari Redis            │
│  - Eksekusi AI Agent & LangChain Nodes │
│  - Window Buffer Memory (Key: chat.id) │
└───────────┬────────────────────────────┘
            │
    ┌───────┴──────────────────────────────┐
    ▼                                      ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ Qdrant Vector DB        │    │ Google Gemini API       │
│ Collection:             │    │ Model:                  │
│  `knowledge_base`       │    │  `gemini-1.5-flash`     │
│ Embeddings:             │    │  (Google AI Studio)     │
│  all-MiniLM-L6-v2 (HF)  │    └─────────────────────────┘
└─────────────────────────┘
```

---

## ⚡ Fitur Utama

- **Zero Software Cost (Rp0):** Seluruh stack menggunakan komponen open-source dan API gratis (OpenRouter `:free` model, Hugging Face Inference API, Qdrant free tier/lokal).
- **Multi-User Isolation (FR-8):** Riwayat percakapan antar user atau grup diisolasi secara ketat menggunakan session key dinamis `{{ $json.message.chat.id }}`.
- **Queue Mode Scalability (NFR-6, NFR-7):** Pemisahan `n8n-main` dan `n8n-worker` dengan Redis + PostgreSQL. Menambah kapasitas paralel cukup dengan `--scale n8n-worker=N`.
- **Anti-Halusinasi Teruji (FR-6):** System prompt AI Agent dirancang tegas untuk hanya menjawab berbasis data dokumen perkuliahan dan menolak berhalusinasi jika konteks tidak ditemukan.
- **Admin Document Ingestion (FR-9):** Workflow upload dokumen via Telegram dilengkapi proteksi IF whitelist `chat_id` administrator.
- **Batch Processing 12 Modul Kuliah:** Skrip Python otomatis untuk mengekstrak, memotong (*chunking*), dan meng-embed 12 modul PDF Semantic Web di folder `Sumber_Data`.

---

## 📂 Struktur Repositori

```
Tugas_RAG/
├── PRD_RAG_Bot_Telegram_n8n.md    # Dokumen Spesifikasi Produk (PRD)
├── docker-compose.yml             # Orkestrasi Docker (Postgres, Redis, n8n-main, n8n-worker, Qdrant, cloudflared)
├── .env.example                   # Template konfigurasi environment
├── .env                           # Konfigurasi aktif lokal
├── requirements.txt               # Pustaka Python untuk pipeline embedding
├── Sumber_Data/                   # Folder 12 file PDF materi Semantic Web
│   ├── 0.0 Overview.pdf
│   ├── 0.1 Introduction.pdf
│   ├── 1. Riset.pdf
│   ├── 2.0 SWTech.pdf
│   ├── 2.1 AP-RDF.pdf
│   ├── 3.0 SWTech - Ontology.pdf
│   ├── 3.1 Protege.pdf
│   ├── 4. Ontology Engineering.pdf
│   ├── 5. Knowledge Base.pdf
│   ├── 6. SPARQL.pdf
│   ├── 7. Reasoning.pdf
│   └── 8. SWRL.pdf
├── workflows/                     # Definisi Workflow n8n siap import
│   ├── workflow_1_ingestion.json  # Workflow Ingestion Dokumen (Admin Whitelist)
│   └── workflow_2_chat_bot.json   # Workflow RAG Chat Bot Telegram (Multi-User)
└── scripts/                       # Skrip automasi & verifikasi
    ├── embed_sumber_data.py       # Ekstraksi, chunking & batch embedding ke Qdrant
    └── test_rag_pipeline.py       # Pengujian retrieval & koneksi vector store
```

---

## 🔑 Kebutuhan Sistem & Akun API

Sebelum menjalankan, siapkan akun dan API key gratis berikut:

| No | Layanan | Fungsi | URL Registrasi |
|---|---|---|---|
| 1 | **Telegram BotFather** | Membuat bot dan mendapatkan Token API Bot | Chat `@BotFather` di Telegram |
| 2 | **User Info Bot** | Mendapatkan Telegram Chat ID admin Anda | Chat `@userinfobot` di Telegram |
| 3 | **Google AI Studio (Gemini)** | Akses model LLM `gemini-1.5-flash` gratis | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| 4 | **Hugging Face** | Token akses untuk embedding `all-MiniLM-L6-v2` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| 5 | **Cloudflare Tunnel** *(Opsional/VPS)* | Ekspos webhook n8n ke HTTPS publik secara gratis | [dash.cloudflare.com](https://dash.cloudflare.com/) |

---

## 🚀 Langkah Instalasi & Deployment

### 1. Salin dan Lengkapi File `.env`
Buka file `.env` di root direktori dan sesuaikan nilainya:
```ini
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstuVWxyz
TELEGRAM_ADMIN_CHAT_ID=987654321
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxx
```

### 2. Jalankan Stack Docker Compose
Jalankan seluruh service (PostgreSQL, Redis, Qdrant, n8n-main, n8n-worker):
```powershell
docker compose up -d
```
Cek status kontainer:
```powershell
docker compose ps
```
Akses editor n8n melalui browser: `http://localhost:5678`

---

## 📚 Ingestion & Batch Embedding 12 Dokumen PDF

Untuk mengindeks seluruh 12 dokumen materi kuliah Semantic Web di `Sumber_Data` ke Qdrant:

### 1. Pasang Dependensi Python
```powershell
pip install -r requirements.txt
```

### 2. Eksekusi Skrip Batch Ingestion
```powershell
python scripts/embed_sumber_data.py
```
Skrip ini akan secara otomatis:
1. Membaca 12 file PDF perkuliahan (termasuk penanganan fallback otomatis).
2. Memecah teks menjadi ~274 chunk (~800–1000 karakter, overlap 100 karakter).
3. Menghasilkan vector embedding 384 dimensi (`sentence-transformers/all-MiniLM-L6-v2`).
4. Mengunggah chunk + metadata (sumber file, nomor halaman, cuplikan) ke collection `knowledge_base` di Qdrant.
5. Melakukan uji similarity search verifikasi.

### 3. Cek Status Pipeline RAG
```powershell
python scripts/test_rag_pipeline.py
```

---

## 🔄 Import & Setup Workflow di n8n

### Langkah 1: Buat Credentials di n8n
Buka dashboard n8n (`http://localhost:5678`), masuk ke menu **Credentials** -> **Add Credential**:
1. **Telegram API**: Masukkan `TELEGRAM_BOT_TOKEN`.
2. **Google Gemini (PaLM) API**:
   - Pilih jenis: `Google Gemini (PaLM) API`
   - Masukkan `GEMINI_API_KEY` (dari [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)).
3. **Qdrant API**:
   - URL: `http://qdrant:6333` (atau URL Qdrant Cloud jika pakai cloud)
   - API Key: (Kosongkan jika Qdrant lokal tanpa auth)
4. **Hugging Face API**:
   - Isi `HUGGINGFACE_API_KEY`

### Langkah 2: Import Workflow
1. Buka menu **Workflows** -> Klik tombol **Add Workflow** -> Pilih menu titik tiga **Import from File...**
2. Import `workflows/workflow_1_ingestion.json` (Workflow Ingestion Admin).
3. Import `workflows/workflow_2_chat_bot.json` (Workflow RAG Chat Bot Telegram).
4. Hubungkan credential yang sudah dibuat pada masing-masing node.
5. Aktifkan saklar **Active** pada kedua workflow.

---

## 🧪 Panduan Pengujian (Multi-User & RAG)

### Uji Coba 1: Uji Relevansi Dokumen (Anti-Halusinasi)
Kirim pesan berikut ke Bot Telegram Anda:
- **Pertanyaan 1 (Ada di materi):**
  > *"Apa itu Resource Description Framework (RDF) dan bagaimana konsep Subject-Predicate-Object bekerja?"*
  - **Hasil yang Diharapkan:** Bot menjelaskan konsep RDF Triple secara akurat mengutip materi perkuliahan.
- **Pertanyaan 2 (Ada di materi):**
  > *"Bagaimana sintaks dasar query SPARQL untuk mencari data?"*
  - **Hasil yang Diharapkan:** Bot memberikan contoh query `SELECT ... WHERE { ?s ?p ?o }` berdasarkan materi SPARQL.
- **Pertanyaan 3 (Di luar materi / Jebakan Halusinasi):**
  > *"Bagaimana resep membuat kue martabak manis keju?"*
  - **Hasil yang Diharapkan:** Bot merespons:
    > *"Maaf, informasi mengenai hal tersebut tidak ditemukan dalam materi perkuliahan yang tersedia."*

### Uji Coba 2: Uji Isolasi Memori Multi-User (FR-8)
1. **User A**: Kirim pesan *"Nama saya Budi, saya sedang mempelajari Ontologi Protege."*
2. **User B**: Kirim pesan *"Siapa nama saya?"*
   - Bot ke User B: *"Maaf, saya belum mengetahui nama Anda."* (Histori tidak bocor ke user lain).
3. **User A**: Kirim pesan *"Siapa nama saya dan materi apa yang saya pelajari tadi?"*
   - Bot ke User A: *"Nama Anda adalah Budi dan Anda sedang mempelajari Ontologi Protege."* (Window Memory User A tersimpan dengan kunci `chat.id`).

### Uji Coba 3: Uji Whitelist Admin Ingestion (FR-9)
1. User biasa (non-admin) mencoba mengirim file PDF ke bot.
   - Respon bot: `⛔ Akses Ditolak: Hanya Administrator terdaftar yang diizinkan mengunggah dokumen baru ke Knowledge Base bersama.`
2. Admin terdaftar mengirim file PDF.
   - Respon bot: `✅ Ingestion Berhasil! Dokumen telah berhasil di-chunk dan di-embed ke Qdrant.`

---

## 📈 Scaling & Manajemen Operasional

### Menambah Kapasitas Worker (Scale Horizontal)
Jika volume user Telegram bertambah tinggi, n8n worker dapat langsung di-scale secara horizontal:
```powershell
docker compose up -d --scale n8n-worker=3
```
Perintah ini akan menjalankan 3 worker n8n secara paralel yang memproses antrian job dari Redis secara concurrent tanpa berebut resource.

### Melihat Log Eksekusi
- Log n8n main: `docker compose logs -f n8n-main`
- Log n8n worker: `docker compose logs -f n8n-worker`
- Log antrian redis: `docker compose logs -f redis`
