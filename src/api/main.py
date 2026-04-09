"""FastAPI entry point for the Aurora Forecast web application."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import forecast_route, models, predictions, snapshot

app = FastAPI(
    title="Aurora Forecast API",
    description="Geomagnetic storm prediction visualisation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(snapshot.router, prefix="/api")
app.include_router(forecast_route.router, prefix="/api")

# In production, serve the built React SPA from the dist directory.
# In development the Vite dev server handles this instead.
_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")


    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(_DIST / "index.html")
