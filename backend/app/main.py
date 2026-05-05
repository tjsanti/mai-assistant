from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.rag.loaders import SUPPORTED_SUFFIXES
from app.runtime import AppRuntime, get_runtime
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUploadIngestResponse,
    HealthResponse,
    IngestResponse,
)

RuntimeDependency = Annotated[AppRuntime, Depends(get_runtime)]

app = FastAPI(title="Personal Knowledge Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ingest", response_model=IngestResponse)
def ingest(runtime: RuntimeDependency) -> IngestResponse:
    return runtime.ingestion_service().run()


@app.post("/documents/upload", response_model=DocumentUploadIngestResponse)
async def upload_documents(
    runtime: RuntimeDependency,
    files: Annotated[list[UploadFile], File()],
) -> DocumentUploadIngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No documents were uploaded.")

    docs_dir = runtime.settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    filenames = [_clean_upload_filename(upload.filename) for upload in files]
    for filename in filenames:
        _validate_supported_document(filename)

    saved_files: list[str] = []
    destinations: list[Path] = []
    reserved_names: set[str] = set()
    for filename in filenames:
        destination = _unused_destination(docs_dir, filename, reserved_names)
        reserved_names.add(destination.name)
        destinations.append(destination)

    for upload, destination in zip(files, destinations, strict=True):
        contents = await upload.read()
        destination.write_bytes(contents)
        saved_files.append(destination.name)

    ingest_result = runtime.ingestion_service().run()
    return DocumentUploadIngestResponse(
        documents_indexed=ingest_result.documents_indexed,
        chunks_indexed=ingest_result.chunks_indexed,
        uploaded_files=saved_files,
    )


def _validate_supported_document(filename: str) -> None:
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type for '{filename}'.",
        )


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runtime: RuntimeDependency,
) -> ChatResponse:
    return runtime.chat_agent().run(request)


def _clean_upload_filename(filename: str | None) -> str:
    cleaned = Path(filename or "").name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Uploaded documents must have filenames.")
    return cleaned


def _unused_destination(docs_dir: Path, filename: str, reserved_names: set[str]) -> Path:
    candidate = docs_dir / filename
    if not candidate.exists() and candidate.name not in reserved_names:
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        candidate = docs_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists() and candidate.name not in reserved_names:
            return candidate
        counter += 1
