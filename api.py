from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from pathlib import Path
from dataclasses import asdict
import os

from analyzer import analyze_audio

app = FastAPI(
    title="Black Red Line AUDIO ANALYZER",
    version="1.0.0",
    description="Audio timing analysis service for DIRECTOR."
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "black-red-line-audio-analyzer", "version": "1.0.0"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), max_sync_points: int = 24):
    suffix = Path(file.filename or "audio.bin").suffix
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)
            tmp_path = tmp.name
        result = analyze_audio(tmp_path, max_sync_points=max_sync_points)
        payload = asdict(result)
        payload["source_file"] = file.filename or payload["source_file"]
        return JSONResponse(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
