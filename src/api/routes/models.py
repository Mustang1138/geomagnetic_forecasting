"""GET /api/models — returns available models with their evaluation metrics."""

from fastapi import APIRouter

from src.api.data_loader import get_metrics

router = APIRouter()


@router.get("/models")
def list_models():
    """Return all models with test-set metrics."""
    return get_metrics()
