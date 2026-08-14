@echo off
cd /d %~dp0..
call cvce\Scripts\activate.bat
python tools\generate_voice_segments.py
pause
