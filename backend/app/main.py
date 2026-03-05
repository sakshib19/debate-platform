import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables
from app.auth import router as auth_router
from app.routes.debates import router as debates_router
from app.routes.results import router as results_router
from app.routes.qa import router as qa_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version="0.1.0")

    # CORS — allow React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create tables on startup (simple approach for dev)
    create_tables()

    # Create required directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RAG_DOCS_DIR, exist_ok=True)

    # Register routers
    app.include_router(auth_router, prefix="/api")
    app.include_router(debates_router, prefix="/api")
    app.include_router(results_router, prefix="/api")
    app.include_router(qa_router, prefix="/api")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    return app


app = create_app()
