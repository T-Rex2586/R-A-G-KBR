@echo off
chcp 65001 >nul
title KBR Bot - Interactive RAG Chat
cls
echo ====================================================================
echo     KBR BOT - ASISTEN AI RAG BERBASIS KNOWLEDGE BASE
echo ====================================================================
echo.
echo Memeriksa dependensi dan memulai Web Server...
echo.

set PYTHONIOENCODING=utf-8
python scripts\start_chat_app.py

pause
