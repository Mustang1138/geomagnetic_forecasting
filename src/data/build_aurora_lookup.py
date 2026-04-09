"""
Build the aurora visibility lookup table from historical OMNI2 data.

For each country above ~45° geomagnetic latitude (north or south), this
script computes the fraction of historical hourly timesteps at which the
auroral oval reached that country's equatorward geomagnetic boundary,
binned by Storm Severity Index (SSI).

The result is a JSON lookup table used by the frontend to colour countries
by their historically-grounded aurora visibility likelihood at a given
predicted SSI level.

Methodology
-----------
The equatorward auroral boundary is estimated from the Kp index using the
NOAA SWPC empirical formula:

    auroral_lat_geomag = 66 - 2 * Kp   (degrees, geomagnetic)

This formula is sourced directly from NOAA's Space Weather Prediction
Centre (swpc.noaa.gov/content/tips-viewing-aurora) and is consistent with
the Feldstein-Starkov (1970) auroral oval model, providing an independent
estimate of auroral extent that does not depend on the project's SSI
pipeline.

Country geomagnetic latitudes are derived from IGRF-based coordinate
conversions and represent the southernmost (northern hemisphere) or
northernmost (southern hemisphere) geomagnetic latitude of each country.
Only countries above roughly 45° geomagnetic latitude in either hemisphere
are included, as aurora is extremely rare below this threshold.

Note: Kp is a 3-hourly index; each 3-hour block is repeated for each
constituent hour in the OMNI2 hourly dataset. Forward-filling is applied
to handle any remaining gaps.

Usage
-----
Run from the project root after the data pipeline has been executed:

    python -m src.data.build_aurora_lookup

Output
------
    frontend/public/aurora_visibility.json

References
----------
- NOAA SWPC (2024) — Kp/auroral latitude relationship
- Feldstein & Starkov (1970) — auroral oval model
- IGRF-13 (2020) — geomagnetic coordinate system
"""

import json
from pathlib import Path

import pandas as pd

from src.utils import load_config, setup_logging

logger = setup_logging()


# Kp → auroral latitude formula (NOAA SWPC)

def kp_to_auroral_lat(kp: pd.Series) -> pd.Series:
    """Convert Kp index to equatorward auroral boundary (geomagnetic degrees).

    Uses the NOAA SWPC empirical formula: lat = 66 - 2*Kp.
    Valid for Kp in [0, 9], giving boundaries from 66° down to 48°.

    Args:
        kp: Series of Kp values (0–9 scale).

    Returns:
        Series of geomagnetic latitudes in degrees.
    """
    return 66.0 - 2.0 * kp


# Country geomagnetic latitude table

# Keyed by GU_A3 (GeoUnit alpha-3) to match the actual unit identifier used
# in the ne_50m_admin_0_map_units GeoJSON. This dataset splits some sovereign
# nations into sub-national units (e.g. GBR → SCT, ENG, WLS, NIR; Belgium →
# BFR, BWR, BCR) so we must key by GU_A3, not ADM0_A3.
#
# geomag_lat: southernmost geomagnetic latitude for NH countries (aurora
#             reaches the country when auroral boundary ≤ this value),
#             northernmost geomagnetic latitude for SH countries (aurora
#             reaches the country when auroral boundary ≥ this value, stored
#             as a negative number).
# hemisphere: 'N' or 'S'
# label: display name for the frontend
#
# Geomagnetic latitudes derived from IGRF-13 coordinate conversions
# (Alken et al., 2021). Values represent the most equatorward point of
# each unit likely to observe aurora, accounting for the ~11° offset
# between geographic and geomagnetic poles.

COUNTRY_GEOMAG = {
    # --- Northern Hemisphere ---

    # British Isles — GBR splits into four units in map_units
    "SCT": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Scotland"},
    "ENG": {"geomag_lat": 50.0, "hemisphere": "N", "label": "England"},
    "WLS": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Wales"},
    "NIR": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Northern Ireland"},
    "IRL": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Ireland"},

    # Scandinavia and Nordic
    "NOR": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Norway"},
    "SWE": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Sweden"},
    "FIN": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Finland"},
    "ISL": {"geomag_lat": 63.0, "hemisphere": "N", "label": "Iceland"},
    "DNK": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Denmark"},
    "FRO": {"geomag_lat": 60.0, "hemisphere": "N", "label": "Faroe Islands"},
    "ALD": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Åland Islands"},

    # Norwegian territories
    "NSV": {"geomag_lat": 71.0, "hemisphere": "N", "label": "Svalbard"},
    "NJM": {"geomag_lat": 68.0, "hemisphere": "N", "label": "Jan Mayen"},

    # Greenland (sovereign under Denmark, separate unit)
    "GRL": {"geomag_lat": 70.0, "hemisphere": "N", "label": "Greenland"},

    # Baltic states
    "EST": {"geomag_lat": 53.0, "hemisphere": "N", "label": "Estonia"},
    "LVA": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Latvia"},
    "LTU": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Lithuania"},

    # Central and Eastern Europe
    # Geomagnetic latitudes represent the northernmost major population centre
    # likely to observe aurora, not the country's southernmost geographic point.
    # The Kp formula reaches a minimum of 48° at Kp=9, so thresholds below
    # 48° will never be reached and are set to 50° as a practical floor.
    "DEU": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Germany"},
    "POL": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Poland"},
    "NLD": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Netherlands"},
    "BFR": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Belgium (Flanders)"},
    "BWR": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Belgium (Wallonia)"},
    "BCR": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Belgium (Brussels)"},
    "CZE": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Czech Republic"},
    "AUT": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Austria"},
    "CHE": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Switzerland"},
    "BLR": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Belarus"},
    "UKR": {"geomag_lat": 50.0, "hemisphere": "N", "label": "Ukraine"},

    # Russia — vast country, aurora regularly visible across Siberia and far north
    "RUS": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Russia"},

    # North America — aurora regularly visible across northern Canada and Alaska
    # 55° covers most of Canada and the northern US states (Minnesota, Michigan etc.)
    "CAN": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Canada"},
    "USA": {"geomag_lat": 55.0, "hemisphere": "N", "label": "United States"},

    #  US States
    "US-AK": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Alaska"},
    "US-MN": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Minnesota"},
    "US-WI": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Wisconsin"},
    "US-MI": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Michigan"},
    "US-ME": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Maine"},
    "US-MT": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Montana"},
    "US-ND": {"geomag_lat": 57.0, "hemisphere": "N", "label": "North Dakota"},
    "US-WA": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Washington"},
    "US-ID": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Idaho"},
    "US-WY": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Wyoming"},
    "US-SD": {"geomag_lat": 55.0, "hemisphere": "N", "label": "South Dakota"},
    "US-VT": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Vermont"},
    "US-NH": {"geomag_lat": 55.0, "hemisphere": "N", "label": "New Hampshire"},
    "US-NY": {"geomag_lat": 54.0, "hemisphere": "N", "label": "New York"},
    "US-OR": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Oregon"},

    # Canadian Provinces
    "CA-YT": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Yukon"},
    "CA-NT": {"geomag_lat": 67.0, "hemisphere": "N", "label": "Northwest Territories"},
    "CA-NU": {"geomag_lat": 70.0, "hemisphere": "N", "label": "Nunavut"},
    "CA-BC": {"geomag_lat": 57.0, "hemisphere": "N", "label": "British Columbia"},
    "CA-AB": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Alberta"},
    "CA-SK": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Saskatchewan"},
    "CA-MB": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Manitoba"},
    "CA-ON": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Ontario"},
    "CA-QC": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Québec"},
    "CA-NL": {"geomag_lat": 58.0, "hemisphere": "N", "label": "Newfoundland and Labrador"},
    "CA-NS": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Nova Scotia"},
    "CA-NB": {"geomag_lat": 55.0, "hemisphere": "N", "label": "New Brunswick"},

    # Russian Regions (aurora-relevant)
    "RU-MUR": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Murmansk"},
    "RU-KR": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Karelia"},
    "RU-ARK": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Arkhangelsk"},
    "RU-NEN": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Nenets"},
    "RU-YAN": {"geomag_lat": 67.0, "hemisphere": "N", "label": "Yamal-Nenets"},
    "RU-KO": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Komi"},
    "RU-KHM": {"geomag_lat": 60.0, "hemisphere": "N", "label": "Khanty-Mansiysk"},
    "RU-KYA": {"geomag_lat": 60.0, "hemisphere": "N", "label": "Krasnoyarsk"},
    "RU-SA": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Sakha (Yakutia)"},
    "RU-KHA": {"geomag_lat": 60.0, "hemisphere": "N", "label": "Khabarovsk"},
    "RU-MAG": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Magadan"},
    "RU-CHU": {"geomag_lat": 65.0, "hemisphere": "N", "label": "Chukchi"},
    "RU-KAM": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Kamchatka"},

    # Additional Russian Oblasts (aurora-relevant)
    "RU-TOM": {"geomag_lat": 58.0, "hemisphere": "N", "label": "Tomsk"},
    "RU-VLG": {"geomag_lat": 58.0, "hemisphere": "N", "label": "Vologda"},
    "RU-ARK": {"geomag_lat": 62.0, "hemisphere": "N", "label": "Arkhangelsk"},
    "RU-KIR": {"geomag_lat": 58.0, "hemisphere": "N", "label": "Kirov"},
    "RU-PER": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Perm"},
    "RU-SVE": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Sverdlovsk"},
    "RU-KGN": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Kurgan"},
    "RU-OMS": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Omsk"},
    "RU-TYU": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Tyumen"},
    "RU-NVS": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Novosibirsk"},
    "RU-IRK": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Irkutsk"},
    "RU-BU": {"geomag_lat": 56.0, "hemisphere": "N", "label": "Buryat"},
    "RU-AMU": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Amur"},
    "RU-SAK": {"geomag_lat": 57.0, "hemisphere": "N", "label": "Sakhalin"},

    # Additional US States (minor aurora relevance)
    "US-MA": {"geomag_lat": 53.0, "hemisphere": "N", "label": "Massachusetts"},
    "US-PA": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Pennsylvania"},
    "US-OH": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Ohio"},
    "US-IL": {"geomag_lat": 53.0, "hemisphere": "N", "label": "Illinois"},
    "US-IN": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Indiana"},
    "US-IA": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Iowa"},
    "US-NE": {"geomag_lat": 54.0, "hemisphere": "N", "label": "Nebraska"},
    "US-WV": {"geomag_lat": 51.0, "hemisphere": "N", "label": "West Virginia"},
    "US-VA": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Virginia"},
    "US-NJ": {"geomag_lat": 51.0, "hemisphere": "N", "label": "New Jersey"},
    "US-CT": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Connecticut"},
    "US-RI": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Rhode Island"},
    "US-DE": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Delaware"},
    "US-MD": {"geomag_lat": 51.0, "hemisphere": "N", "label": "Maryland"},
    "US-MO": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Missouri"},
    "US-KS": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Kansas"},
    "US-NV": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Nevada"},
    "US-UT": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Utah"},
    "US-CO": {"geomag_lat": 52.0, "hemisphere": "N", "label": "Colorado"},

    # Canadian Province
    "CA-PE": {"geomag_lat": 55.0, "hemisphere": "N", "label": "Prince Edward Island"},

    # Australian States
    "AU-TAS": {"geomag_lat": -55.0, "hemisphere": "S", "label": "Tasmania"},
    "AU-VIC": {"geomag_lat": -53.0, "hemisphere": "S", "label": "Victoria"},
    "AU-NSW": {"geomag_lat": -52.0, "hemisphere": "S", "label": "New South Wales"},
    "AU-SA": {"geomag_lat": -50.0, "hemisphere": "S", "label": "South Australia"},
    "AU-WA": {"geomag_lat": -50.0, "hemisphere": "S", "label": "Western Australia"},

    # outhern Hemisphere
    # geomag_lat stored as negative degrees (south).
    # Southern geomagnetic pole offset means SH aurora visible at higher
    # geographic latitudes than NH equivalent — thresholds adjusted accordingly.
    "NZL": {"geomag_lat": -55.0, "hemisphere": "S", "label": "New Zealand"},
    "AUS": {"geomag_lat": -55.0, "hemisphere": "S", "label": "Australia"},
    "ARG": {"geomag_lat": -52.0, "hemisphere": "S", "label": "Argentina"},
    "CHL": {"geomag_lat": -52.0, "hemisphere": "S", "label": "Chile"},
    "ZAF": {"geomag_lat": -50.0, "hemisphere": "S", "label": "South Africa"},
    "FLK": {"geomag_lat": -58.0, "hemisphere": "S", "label": "Falkland Islands"},
    "SGS": {"geomag_lat": -60.0, "hemisphere": "S", "label": "South Georgia"},
}

# SSI bins

# 10 bins covering [0, 0.50] plus a catch-all for > 0.50.
# Bin edges chosen to align with the SSI severity thresholds used elsewhere
# in the pipeline and frontend.
SSI_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.01]
SSI_BIN_LABELS = [
    "0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20",
    "0.20-0.25", "0.25-0.30", "0.30-0.35", "0.35-0.40",
    "0.40-0.45", "0.45-0.50", ">0.50",
]


# Core computation

def _aurora_reaches_country(
        auroral_lat: pd.Series,
        country_geomag_lat: float,
        hemisphere: str,
) -> pd.Series:
    """Return a boolean mask: True when aurora reaches the given country.

    For northern hemisphere countries the auroral boundary must move
    equatorward to reach the country's southernmost point, so we check
    whether the boundary latitude is ≤ the country's geomagnetic latitude.

    For southern hemisphere countries the geometry is mirrored — the
    boundary must move to lower absolute latitudes (i.e. more equatorward
    in the south), so we check whether the boundary is ≥ the absolute
    value of the country's (negative) geomagnetic latitude.

    Args:
        auroral_lat: Series of equatorward auroral boundary latitudes
            in geomagnetic degrees (positive = north).
        country_geomag_lat: The country's boundary geomagnetic latitude.
        hemisphere: 'N' or 'S'.

    Returns:
        Boolean Series.
    """
    if hemisphere == "N":
        return auroral_lat <= country_geomag_lat
    else:
        # For SH, auroral_lat from kp_to_auroral_lat gives the northern
        # boundary — mirror it for southern hemisphere by negating.
        southern_boundary = -auroral_lat
        return southern_boundary >= abs(country_geomag_lat)


def build_lookup(processed_dir: Path) -> dict:
    """Compute per-country, per-SSI-bin aurora visibility fractions.

    Loads all three data splits (train, val, test) and concatenates them
    to maximise the historical record used for the empirical fractions.
    SSI values are inverse-scaled back to physical units [0, 1] before
    binning, since the baseline CSVs store scaled (z-normalised) values.

    Args:
        processed_dir: Path to the processed data directory containing
            train/val/test baseline CSVs and scaler_y.pkl.

    Returns:
        Nested dict: { GU_A3: { ssi_bin_label: fraction, ..., meta } }
    """
    import pickle

    # Load and concatenate all three splits for maximum coverage
    frames = []
    for split in ("train", "val", "test"):
        csv_path = processed_dir / f"{split}_baseline.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path, parse_dates=["datetime"]))
            logger.info("Loaded %s (%d rows).", csv_path.name, len(frames[-1]))
        else:
            logger.warning("%s not found — skipping.", csv_path.name)

    if not frames:
        raise FileNotFoundError(
            "No baseline CSV files found in processed_dir. "
            "Run the full data pipeline first."
        )

    df = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    required = {"kp", "storm_severity_index", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed CSV is missing required columns: {missing}. "
            "Ensure parsers.py has been updated to include Kp and the "
            "data pipeline has been re-run."
        )

    # Inverse-scale SSI from z-score space back to physical [0, 1] values.
    # The baseline CSVs store scaled targets; binning requires physical units.
    scaler_path = processed_dir / "scaler_y.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"scaler_y.pkl not found at {scaler_path}. "
            "Run the preprocessing pipeline first."
        )
    with open(scaler_path, "rb") as fh:
        scaler_y = pickle.load(fh)

    df["ssi_physical"] = scaler_y.inverse_transform(
        df["storm_severity_index"].values.reshape(-1, 1)
    ).flatten()

    # Clip to valid range — inverse scaling can produce slight extrapolations
    df["ssi_physical"] = df["ssi_physical"].clip(0.0, 1.0)

    # Forward-fill Kp gaps — Kp is a 3-hourly index repeated per hour;
    # remaining NaNs after parsing are genuine missing data periods.
    df["kp"] = df["kp"].ffill().bfill()
    df = df.dropna(subset=["kp", "ssi_physical"])

    logger.info("Computing auroral boundaries for %d total timesteps …", len(df))

    # Compute equatorward auroral boundary from Kp
    df["auroral_lat_geomag"] = kp_to_auroral_lat(df["kp"])

    # Assign SSI bins using physical (inverse-scaled) values
    df["ssi_bin"] = pd.cut(
        df["ssi_physical"],
        bins=SSI_BINS,
        labels=SSI_BIN_LABELS,
        include_lowest=True,
        right=False,
    )

    # Count total timesteps per SSI bin (denominator)
    bin_counts = df["ssi_bin"].value_counts()

    lookup = {}

    for adm0_a3, meta in COUNTRY_GEOMAG.items():
        geomag_lat = meta["geomag_lat"]
        hemisphere = meta["hemisphere"]

        # Boolean mask: aurora reaches this country at each timestep
        reaches = _aurora_reaches_country(
            df["auroral_lat_geomag"], geomag_lat, hemisphere
        )

        # Compute fraction per SSI bin
        bin_fractions = {}
        for bin_label in SSI_BIN_LABELS:
            in_bin = df["ssi_bin"] == bin_label
            n_total = int(bin_counts.get(bin_label, 0))

            if n_total == 0:
                bin_fractions[bin_label] = 0.0
            else:
                n_visible = int((in_bin & reaches).sum())
                bin_fractions[bin_label] = round(n_visible / n_total, 4)

        lookup[adm0_a3] = {
            "label": meta["label"],
            "hemisphere": hemisphere,
            "geomag_lat": geomag_lat,
            "visibility": bin_fractions,
        }

        logger.info(
            "  %s (%s): max visibility %.1f%% at highest SSI bin.",
            meta["label"],
            adm0_a3,
            max(bin_fractions.values()) * 100,
        )

    return lookup


# Entry point

def main() -> None:
    """Build and save the aurora visibility lookup table."""
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    lookup = build_lookup(processed_dir)

    # Write to frontend/public so it's served as a static asset
    output_path = Path("frontend/public/aurora_visibility.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        json.dump(lookup, fh, indent=2)

    logger.info(
        "Aurora visibility lookup saved → %s (%d countries).",
        output_path,
        len(lookup),
    )


if __name__ == "__main__":
    main()
