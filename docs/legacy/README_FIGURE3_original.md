# Independent reproduction of Li et al. (2021) Figure 3

Run or resume the four seventh-order PDM simulations and export Figure 3:

    python -m esf_sim reproduce-figure3 --config configs/figure3.json

Rebuild the figure and reports from completed snapshots:

    python -m esf_sim reproduce-figure3 --config configs/figure3.json --skip-simulation

The four panels use alpha values 0.2, 0.4, 0.6, and 0.8 at 3150, 3150,
3050, and 2860 s, respectively. All final panels come from independent numerical
solutions. The ScienceDB arrays are used only for the acceptance report.

The official configuration uses the active point-centered equation (2.21) baseline with equal-value vertical potential extrapolation. Results are written below `output/best_baseline/figure3`. The promoted alpha=0.4 snapshot is already populated; the other alpha panels are generated or resumed as needed. They include stable density snapshots,
archived configurations, checkpoints and logs, a combined source-data archive,
JSON metrics, and SVG/PDF/PNG/TIFF figure files. Raster files are exported at
600 dpi, while PDF and SVG retain editable text.

The previous point-centered zero-potential boundary is preserved with:

    python -m esf_sim reproduce-figure3 --config configs/figure3_legacy_point_centered_dirichlet.json

The older conservative harmonic-face baseline is preserved with:

    python -m esf_sim reproduce-figure3 --config configs/figure3_legacy_harmonic.json
