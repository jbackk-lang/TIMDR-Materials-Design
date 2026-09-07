@echo off
REM run.bat - uruchamia API TIMDR-Materials-Design lokalnie na Windows.
REM
REM Co robi:
REM   1. Tworzy lokalne srodowisko wirtualne .venv (jesli jeszcze nie istnieje).
REM   2. Instaluje/aktualizuje zaleznosci z requirements.txt.
REM   3. Startuje serwer API (uvicorn) na http://127.0.0.1:8000
REM      - dokumentacja Swagger: http://127.0.0.1:8000/docs
REM
REM Jesli padnie .venv cd "C:\Users\twoja sciezka\TIMDR-Materials-Design" dalej
REM polecenie z terminala: Remove-Item -Recurse -Force .venv
REM
REM Wymaga Pythona 3.10+ dostepnego w PATH jako "python".
REM Zatrzymanie serwera: Ctrl+C w tym oknie.

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [BLAD] Nie znaleziono "python" w PATH. Zainstaluj Pythona 3.10+ z python.org
    echo        i zaznacz "Add python.exe to PATH" podczas instalacji.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Tworze srodowisko wirtualne .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [BLAD] Nie udalo sie utworzyc .venv
        pause
        exit /b 1
    )
) else (
    echo [1/3] .venv juz istnieje, pomijam tworzenie.
)

echo [2/3] Instaluje zaleznosci z requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [BLAD] Instalacja zaleznosci nie powiodla sie.
    pause
    exit /b 1
)

echo [3/3] Startuje API na http://127.0.0.1:8000  (dokumentacja: /docs)
echo        Zatrzymanie: Ctrl+C
".venv\Scripts\python.exe" -m uvicorn material_timdr.api:app --host 127.0.0.1 --port 8000

endlocal
