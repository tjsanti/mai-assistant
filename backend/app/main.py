from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.runtime import AppRuntime, get_runtime
from app.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse

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


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runtime: RuntimeDependency,
) -> ChatResponse:
    return runtime.chat_agent().run(request)
