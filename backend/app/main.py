from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import AgentGraph
from app.config import get_settings
from app.rag.ingestion import IngestionService
from app.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse

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
def ingest() -> IngestResponse:
    service = IngestionService.from_settings(get_settings())
    return service.run()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    graph = AgentGraph.from_settings(get_settings())
    return graph.run(request)

