#!/usr/bin/env python3
"""
Activate Workflows Script
Mengaktifkan dan mempublikasikan seluruh workflow n8n (Queue Mode) secara otomatis,
sehingga Telegram Bot dan Webhook aktif 24/7 tanpa perlu menekan tombol eksekusi manual di canvas editor.
"""

import subprocess
import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout.strip(), res.stderr.strip(), res.returncode
    except Exception as e:
        return "", str(e), 1

def activate():
    print("=" * 60)
    print("   AKTIVASI OTOMATIS WORKFLOW N8N (PRODUKSI 24/7)")
    print("=" * 60)

    # 1. Ambil list workflow dari database PostgreSQL
    print("[*] Mengambil daftar workflow dari PostgreSQL n8n...")
    sql = "SELECT id, name, active FROM workflow_entity;"
    out, err, code = run_cmd(f'docker exec rag-postgres psql -U n8n -d n8n -t -A -c "{sql}"')
    if code != 0:
        print(f"[!] Gagal mengambil workflow: {err}")
        return

    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        print("[!] Belum ada workflow yang terdaftar di n8n.")
        return

    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3:
            w_id, w_name, w_active = parts[0], parts[1], parts[2]
            print(f"\n[+] Workflow: '{w_name}' (ID: {w_id})")
            print(f"    Status saat ini: {'Aktif (True)' if w_active == 't' else 'Nonaktif (False)'}")

            # Publikasikan & aktifkan
            print(f"    [*] Mengaktifkan & mempublikasikan workflow {w_id}...")
            p_out, p_err, p_code = run_cmd(f"docker exec rag-n8n-main n8n publish:workflow --id={w_id}")
            if p_code == 0:
                print(f"    [✓] Berhasil dipublikasikan!")
            else:
                print(f"    [!] Catatan: {p_err or p_out}")

    # 2. Restart container n8n-main agar listener aktif
    print("\n[*] Merestart container rag-n8n-main agar semua trigger aktif...")
    r_out, r_err, r_code = run_cmd("docker restart rag-n8n-main")
    if r_code == 0:
        print("[✓] rag-n8n-main berhasil direstart.")
    else:
        print(f"[!] Gagal restart n8n-main: {r_err}")

    # 3. Verifikasi Webhook Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token:
        print("\n[*] Memeriksa status Webhook Telegram Bot...")
        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getWebhookInfo", timeout=10)
            data = r.json()
            if data.get("ok"):
                url = data.get("result", {}).get("url", "")
                if url:
                    print(f"[✓] Webhook Telegram TERDAFTAR AKTIF!")
                    print(f"    URL: {url}")
                    print(f"    Pending Updates: {data['result'].get('pending_update_count', 0)}")
                    print("\n🎉 Sukses! Anda sekarang bisa chat langsung di Telegram tanpa perlu menekan 'Execute workflow' manual!")
                else:
                    print("[!] Webhook URL masih kosong. Pastikan workflow Telegram sudah terpasang trigger valid.")
            else:
                print(f"[!] Telegram API error: {data}")
        except Exception as e:
            print(f"[!] Error saat menghubungi Telegram API: {e}")

if __name__ == "__main__":
    activate()
