"""Generate every dissertation figure in one pass.

Usage::

    python -m src.evaluation.figures
"""

from src.evaluation.figures._common import (
    METRICS_DIR,
    OUT_DIR,
    PLOTS_DIR,
    load_predictions,
)
from src.evaluation.figures.diagnostics import (
    fig_dst_vs_ssi,
    fig_solar_cycle_timeline,
    fig_ssi_acf,
    fig_ssi_class_distribution,
)
from src.evaluation.figures.loss_curves import fig_loss_curves
from src.evaluation.figures.main_results import (
    fig_comprehensive,
    fig_metrics_r2_ss,
    fig_metrics_rmse_mae,
    fig_scatter_combined,
    fig_timeseries_combined,
)
from src.evaluation.figures.significance import (
    fig_dm_heatmap,
    fig_stratified_skill,
)
from src.evaluation.figures.storm_window import (
    fig_storm_residuals,
    fig_storm_timeseries,
)


def main(skip_loss_curves: bool = False) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_predictions()

    # Headline test-set results
    fig_timeseries_combined(preds, OUT_DIR / "fig_6_1_timeseries_combined.png")
    fig_scatter_combined(preds, OUT_DIR / "fig_6_2_scatter_combined.png")
    fig_metrics_rmse_mae(OUT_DIR / "fig_metrics_rmse_mae.png")
    fig_metrics_r2_ss(OUT_DIR / "fig_metrics_r2_ss.png")
    fig_comprehensive(preds, OUT_DIR / "fig_comprehensive.png")

    # Storm-window detail
    fig_storm_timeseries(preds, OUT_DIR / "fig_storm_timeseries.png")
    fig_storm_residuals(preds, OUT_DIR / "fig_storm_residuals.png")

    # Diagnostics
    fig_solar_cycle_timeline(OUT_DIR / "fig_solar_cycle_timeline.png")
    fig_ssi_acf(OUT_DIR / "fig_ssi_acf.png")
    fig_dst_vs_ssi(OUT_DIR / "fig_dst_vs_ssi.png")
    fig_ssi_class_distribution(OUT_DIR / "ssi_class_distribution.png")

    # Significance and stratified
    fig_dm_heatmap(
        main_csv=METRICS_DIR / "dm_test_main.csv",
        ablation_csv=METRICS_DIR / "dm_test_ablation.csv",
        output_path=PLOTS_DIR / "fig_f3_dm_significance_matrix.png",
    )
    fig_stratified_skill(
        stratified_csv=METRICS_DIR / "stratified_combined.csv",
        output_path=PLOTS_DIR / "fig_6_14_stratified_skill.png",
    )

    # Loss curves last because they retrain LSTM and GRU (slow)
    if not skip_loss_curves:
        fig_loss_curves(OUT_DIR / "fig_loss_curves.png")


if __name__ == "__main__":
    main()
