# Heinzel et al. (2015) Table 1 — calibrated 1–100 Mm extension

This package contains the 1–100 Mm extension constructed from the completed **Promweaver PRD + cone-boundary** grid and calibrated to the published Table 1 at 10, 20, and 30 Mm.

## Recommended final products

- `output/heinzel_calibrated_extension_1_to_100Mm.csv` — tidy master table (2500 rows).
- `output/heinzel_extension_i_by_height.csv` — wide ionization-degree table.
- `output/heinzel_extension_f_by_height_1e16_cm3.csv` — wide f table.
- `output/single_anchor_cross_validation_summary.csv` — independent height-response validation.
- `plots/` — representative height curves.

## Definitions

The primary Promweaver quantities are

`i_pw_H = n_p / n_H`

and

`f_pw_H = n_p^2 / n_2`,

with f reported in units of `1e16 cm^-3`. These are used instead of total-electron quantities because the original Heinzel Table-1 treatment neglects helium ionization in the pressure relation. The raw total-electron quantities are retained for diagnostics.

## Why calibrate?

The modern Promweaver/FAL-C calculation reproduces the published ionization degree closely, but its absolute `f` normalization differs substantially from the 2015 table. Crucially, the *relative height response* agrees very well.

Using only the published 10-Mm values to normalize Promweaver, the model predicts the published values at:

- **20 Mm:** median error 0.47% for i and 1.64% for f.
- **30 Mm:** median error 0.56% for i and 2.98% for f.

This is the empirical justification for using Promweaver to supply the height dependence.

## Final calibration formula

For quantity `X` (i or f), at each `(T,p)` and published anchor height `Ha`:

`C_X(Ha,T,p) = X_Heinzel(Ha,T,p) / X_PW(Ha,T,p)`

and

`X_extension(H,T,p) = C_X(H,T,p) * X_PW(H,T,p)`.

`C_X` is held fixed below 10 Mm and above 30 Mm. Between the published anchors it is interpolated geometrically so that the final table is continuous and exactly reproduces the published 10/20/30-Mm values.

## Rebuild

`build_extension.py` is the original calculation script. It retains absolute
`/mnt/data/` source and output paths and removes its configured output directory
before rebuilding. See the [archive guide](../README.md#historical-builder) before
using it; the supplied outputs can be inspected without running the builder.

## Scientific caution

This is a **model-based calibrated extension**, not a new published non-LTE reference table. It is strongest where the prominence-slab approximation is appropriate. Values at roughly 1–5 Mm should be treated with additional caution.
