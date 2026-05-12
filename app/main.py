from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.db.vector_store import init_schema
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    yield


app = FastAPI(
    title="VisionRAG",
    description="Multimodal Document Intelligence — Extract, Embed, Query",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
