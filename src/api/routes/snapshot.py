"""GET /api/snapshot?idx=142 — returns all model predictions at a single timestep."""

from fastapi import APIRouter, HTTPException, Query

from src.api.data_loader import load_aligned_predictions

router = APIRouter()


@router.get("/snapshot")
def get_snapshot(idx: int = Query(default=0, ge=0, description="Zero-based timestep index")):
    """Return every model's prediction at one timestep."""
    df = load_aligned_predictions()

    if idx >= len(df):
        raise HTTPException(400, detail=f"Index {idx} out of range (max {len(df) - 1}).")

    row = df.iloc[idx]

    return {
        "idx": idx,
        "dt": row["datetime"].strftime("%Y-%m-%d %H:%M"),
        "lat": float(row["auroral_lat"]),
        "cl": row["storm_class"],
        "true": float(row["true"]),
        "models": {k: float(row[k]) for k in ("rf", "lr", "ls", "gr", "pe")},
    }
