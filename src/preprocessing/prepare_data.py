"""Entry point for downloading OMNI2 data and running the full preprocessing pipeline."""

from src.data.data_fetcher import DataLoader, resolve_end_date
from src.preprocessing.preprocess import DataPreprocessor
from src.utils import setup_logging

logger = setup_logging()


def prepare(config_path: str = "config.yaml") -> None:
    """Execute the full data download and preprocessing pipeline."""
    loader = DataLoader(config_path=config_path)
    date_range_cfg = loader.config["data"]["date_range"]

    start_year = int(date_range_cfg["start"][:4])
    end_date = resolve_end_date(date_range_cfg["end"])
    end_year = end_date.year

    logger.info(
        "Preparing data from %d to %s.",
        start_year,
        end_date.strftime("%Y-%m-%d"),
    )

    logger.info("Downloading OMNI2 data (%d–%d) …", start_year, end_year)
    loader.download_omni2_range(start_year, end_year)

    logger.info("Loading and combining OMNI2 files …")
    omni_df = loader.load_omni2_range(start_year, end_year)

    if omni_df.empty:
        logger.error(
            "No OMNI2 data loaded — aborting. "
            "Check that the raw data files exist in '%s'.",
            loader.raw_dir,
        )
        return

    combined_path = loader.raw_dir / "omni2_combined.csv"
    omni_df.to_csv(combined_path, index=False)
    logger.info("Combined dataset saved → %s (%d rows).", combined_path, len(omni_df))

    logger.info("Running preprocessing pipeline …")
    preprocessor = DataPreprocessor(config_path=config_path)
    summary = preprocessor.run(
        input_csv=str(combined_path),
        output_dir=loader.config["data"]["processed_dir"],
    )

    logger.info("Preprocessing complete: %s", summary)


if __name__ == "__main__":
    prepare()
