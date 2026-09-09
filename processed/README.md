# Calibration provenance

[Project README](../README.md)

This directory holds the reference calculation products used to validate the
runtime opacity table. These are small scientific reference assets and belong
in version control.

[`heinzel_table1_extension_final/`](heinzel_table1_extension_final/README.md)
preserves the supplied calibration archive: raw grid, published-reference CSV,
calibrated tables, validation summaries, plots, and the original builder.

| Location | Role |
| :--- | :--- |
| `heinzel_table1_extension_final/data/` | Raw grid and published anchors |
| `heinzel_table1_extension_final/output/` | Calibrated master table and validation products |
| `heinzel_table1_extension_final/plots/` | Archived diagnostic figures |
| `../synthetic_filaments/data/` | Runtime table and spine library packaged with FILOS |

The scientific regression already expects this archive at its current path. It
checks that the packaged table matches the archived master byte for byte and
preserves the published anchors. See [integration details](heinzel_table1_extension_final/PROJECT_INTEGRATION.md).

## Historical builder

`build_extension.py` is preserved as supplied. It uses absolute `/mnt/data/`
source/output paths, imports pandas, and deletes its configured output directory
before rebuilding. It is an archival calculation script, not a portable FILOS
entry point. The supplied products can be inspected and checked without running
it. Its source code has not been changed as part of repository organization.
