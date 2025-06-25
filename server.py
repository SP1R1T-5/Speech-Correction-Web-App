#!/usr/bin/env python3
"""
server.py – FastAPI bridge for Speech-Correction Assistant
Folder layout:

webserver/
├── server.py
├── frontend/
│   └── index.html
└── voicefixer/   (git-cloned; already pip-installed with  -e )
"""

from pathlib import Path
import shutil, subprocess, tempfile, uuid, os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from voicefixer import VoiceFixer
import librosa, soundfile as sf
from pesq import pesq
from pystoi import stoi

# ────── basic paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"
TARGET_SR = 16_000

# ────── FastAPI setup ────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # loosened for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1️⃣  API ENDPOINTS FIRST (so they win the path-match race)
@app.get("/health", include_in_schema=False)
async def health():
    return JSONResponse({"status": "ok"})

@app.post("/process", include_in_schema=False)
async def process_audio(
    audio: UploadFile = File(...),
    text: str | None = Form(None),
    include_metrics: str = Form("false"),
):
    tmpdir = Path(tempfile.mkdtemp())
    raw_path = tmpdir / f"upload_{uuid.uuid4()}{Path(audio.filename).suffix}"
    with raw_path.open("wb") as f:
        f.write(await audio.read())

    wav_in  = tmpdir / "input.wav"
    wav_out = tmpdir / "enhanced.wav"

    try:
        # ── convert WebM/Opus → WAV if needed ─────────────────
        if audio.content_type.startswith("audio/webm") or raw_path.suffix == ".webm":
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", str(TARGET_SR), wav_in],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.move(raw_path, wav_in)

        # ── enhance with VoiceFixer ───────────────────────────
        vf.restore(input=str(wav_in), output=str(wav_out), cuda=False, mode=0)

        # ── metrics (optional) ────────────────────────────────
        headers = {}
        if include_metrics.lower() == "true":
            yo, _ = librosa.load(wav_in, sr=TARGET_SR, mono=True)
            ye, _ = librosa.load(wav_out, sr=TARGET_SR, mono=True)
            m = min(len(yo), len(ye))
            if m >= TARGET_SR // 4:
                yo, ye = yo[:m], ye[:m]
                headers["X-PESQ-Score"] = f"{pesq(TARGET_SR, yo, ye, 'wb'):.3f}"
                headers["X-STOI-Score"] = f"{stoi(yo, ye, TARGET_SR, extended=False):.3f}"

        return FileResponse(wav_out, media_type="audio/wav",
                            filename="enhanced.wav", headers=headers)

    except subprocess.CalledProcessError:
        raise HTTPException(500, "ffmpeg conversion failed")
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# 2️⃣  STATIC FILES (placed *after* the API routes)
if not INDEX_FILE.exists():
    raise RuntimeError(f"index.html not found at {INDEX_FILE}")

# root `/` serves index.html; anything under /static/ serves raw files
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(INDEX_FILE)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ────── Load VoiceFixer once at startup ──────────────────────
print("🔧 Loading VoiceFixer …")
vf = VoiceFixer()
print("✅ VoiceFixer ready")
