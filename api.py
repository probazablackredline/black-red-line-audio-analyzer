from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from analyzer import analyze_audio


# ---------------------------------------------------------
# MCP SERVER
# ---------------------------------------------------------

mcp = MCPServer("Black Red Line AUDIO ANALYZER")


@mcp.tool()
def analyzer_status() -> dict:
    """Check whether the Black Red Line AUDIO ANALYZER is available."""
    return {
        "status": "ok",
        "service": "black-red-line-audio-analyzer",
        "schema_version": "1.1",
    }


# Разрешаем публичный домен Render.
security = TransportSecuritySettings(
    allowed_hosts=[
        "black-red-line-audio-analyzer.onrender.com",
        "black-red-line-audio-analyzer.onrender.com:*",
    ],
)


# ВАЖНО: сначала создаём MCP-приложение.
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    transport_security=security,
)


# ---------------------------------------------------------
# APPLICATION LIFESPAN
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


# ---------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Black Red Line AUDIO ANALYZER",
    version="1.1.0",
    description="Audio timing analysis service for DIRECTOR.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "black-red-line-audio-analyzer",
        "version": "1.1.0",
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    max_sync_points: int = 24,
):
    suffix = Path(file.filename or "audio.bin").suffix
    tmp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

            tmp_path = tmp.name

        result = analyze_audio(
            tmp_path,
            max_sync_points=max_sync_points,
        )

        payload = asdict(result)
        payload["source_file"] = file.filename or payload["source_file"]

        return JSONResponse(payload)

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------
# MCP ENDPOINT
# ---------------------------------------------------------

app.mount("/mcp", mcp_app)
