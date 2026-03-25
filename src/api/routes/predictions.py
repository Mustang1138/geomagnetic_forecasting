"""GET /api/predictions?model=rf – returns the full test-set time series for one model."""

from fastapi import APIRouter, HTTPException, Query

from src.api.data_loader import load_aligned_predictions

router = APIRouter()

VALID_MODELS = {"rf", "lr", "ls", "gr", "pe"}


@router.get("/predictions")
def get_predictions(model: str = Query(default="rf", description="Model key: rf | lr | ls | gr | pe")):
    """
    Return columnar arrays for compact transfer.

        { "model": "rf", "n": 3400, "dt": […], "lat": […], "cl": […], "true": […], "pred": […] }
    """
    if model not in VALID_MODELS:
        raise HTTPException(400, detail=f"Unknown model '{model}'. Valid: {sorted(VALID_MODELS)}")

    df = load_aligned_predictions()

    return {
        "model": model,
        "n": len(df),
        "dt": df["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "lat": df["auroral_lat"].tolist(),
        "cl": df["storm_class"].tolist(),
        "true": df["true"].tolist(),
        "pred": df[model].tolist(),
    }
