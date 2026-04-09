"""GET /api/forecast — returns a 7-day, 6-hourly SSI forecast for all models."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.forecast import generate_forecast

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/forecast")
def get_forecast():
    """Generate and return a 7-day geomagnetic storm severity forecast."""
    result = generate_forecast()

    if result is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast unavailable — could not retrieve real-time DSCOVR "
                "solar wind data from NOAA SWPC. Please try again shortly."
            ),
        )

    return result
