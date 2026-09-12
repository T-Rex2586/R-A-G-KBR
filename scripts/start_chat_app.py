#!/usr/bin/env python3
"""
Launcher Script untuk Semantic Web GPT Web Application
Menjalankan server FastAPI Uvicorn dan membuka browser otomatis.
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

def main():
    print("=" * 65)
    print("   [+] KBR BOT - ASISTEN AI RAG INTERAKTIF")
    print("=" * 65)
    print(" - Basis Data  : Qdrant Vector Store (12 Modul Kuliah, 1.109 Chunks)")
    print(" - Model AI    : Google Gemini Flash Lite")
    print(" - Antarmuka   : Web Chat Interaktif (Modern ChatGPT UI)")
    print("=" * 65)

    port = 8000
    url = f"http://localhost:{port}"

    print(f"\n[*] Membuka browser di {url}...")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(f"[*] Menjalankan server pada port {port}...")
    print(f"[*] Tekan Ctrl+C untuk menghentikan server.\n")

    import uvicorn
    # Jalankan server
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
