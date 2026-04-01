"""GET /api/forecast — returns a 7-day, 6-hourly SSI forecast for all models."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.forecast import generate_forecast

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/forecast")
def get_forecast():
    """Generate and return a 7-day geomagnetic storm severity forecast.

    Fetches real-time DSCOVR solar wind data, seeds all five trained models,
    and runs a 28-step autoregressive forecast in 6-hour blocks.

    Returns a JSON object containing:
        - ``generated_at``: UTC timestamp of forecast generation.
        - ``steps``: Number of forecast steps (28).
        - ``step_hours``: Hours between steps (6).
        - ``frozen_conditions_assumed``: ``true`` — current solar wind
          conditions are held constant across all forecast steps.
        - ``timestamps``: List of forecast timestamp strings.
        - ``models``: Per-model arrays of ``ssi``, ``auroral_lat``,
          and ``storm_class``.

    Raises:
        HTTPException: 503 if the DSCOVR feed is unavailable or the
            seed window cannot be constructed.
    """
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
