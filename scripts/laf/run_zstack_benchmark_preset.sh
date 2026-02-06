#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/zstack_benchmark.py"
POSITIONS_FILE="${REPO_ROOT}/scripts/positions.json"

# Single-site calibration (first position only)
uv run python "${BENCHMARK_SCRIPT}" --microscope "squid-royal-pearl" \
    --cal-range 400 --cal-steps 29 \
    --test-range 400 --test-steps 81 \
    --positions "${POSITIONS_FILE}" \
    --no-display --laf-exposure 5.0

# RESULTS (single-site calibration):
# ========================================================================================================================
# TABLE 2: AVERAGE RESULTS ACROSS 4 POSITIONS
# ========================================================================================================================
# Tracker         Avg MAE   |z|<=15   |z|<=5   MaxErr  Method
# ------------------------------------------------------------------------------------------------------------------------
# CC 2D Spline       3.61     3.34     3.23      6.8  2D CC + spline interp
# CC 2D              3.76     3.38     3.27      7.6  2D cross-corr full img
# CC 2D+JPEG         3.76     3.38     3.27      7.6  2D CC w/ JPEG compression
# CC 1D              5.03     3.57     3.53     15.5  1D cross-corr of profiles
# Centroid          31.33    14.91    14.87     89.1  intensity centroid tracking
# Peak 1D           34.74    29.98    29.76     92.4  track peak in 1D profile
# Gauss 1D          34.75    31.17    31.24     91.8  gaussian fit to 1D profile
# SQUID+s50         39.25    34.50    34.53    132.9  SQUID + gaussian s=50
# CNN               45.08     6.66     5.71    179.0  CNN on 64x64 resized
# Centroid 2D       50.61    95.58    92.23    128.4  centroid + brightness 2D lookup
# SQUID+s20         57.93    70.01    77.88    158.5  SQUID + gaussian s=20
# SQUID+s10         64.13    81.11    80.05    161.4  SQUID + gaussian s=10
# SQUID+s4          82.25    92.26    72.99    223.3  SQUID + gaussian s=4
# SQUID             84.83    88.55    52.59    228.1  real SQUID: blur s=1 + peak
# SQUID+s2          85.77    85.65    52.00    228.8  SQUID + gaussian s=2
# SQUID+s0          85.81    74.05    49.87    245.8  SQUID, no smoothing
# Zero             101.20     7.50     2.50    200.0  always predicts 0
# ========================================================================================================================
#
# ====================================================================================================
# TABLE 3: CALIBRATION SITE vs OTHER SITES (GENERALIZATION)
# ====================================================================================================
# Tracker           Cal MAE   Cal |z|<=15    Other MAE   Other |z|<=15      D MAE   D |z|<=15
# ----------------------------------------------------------------------------------------------------
# CC 2D Spline         0.64         0.77         4.60           4.20      +3.96      +3.43
# CC 2D                0.71         0.67         4.78           4.28      +4.07      +3.61
# CC 2D+JPEG           0.70         0.67         4.78           4.29      +4.08      +3.61
# CC 1D                0.52         0.40         6.54           4.63      +6.01      +4.23
# Centroid            25.51        16.99        33.28          14.21      +7.76      -2.78
# Peak 1D             18.44        32.08        40.17          29.27     +21.73      -2.81
# Gauss 1D            18.49        33.37        40.17          30.44     +21.67      -2.93
# SQUID+s50           17.85        32.61        46.38          35.13     +28.52      +2.52
# CNN                  8.04         6.38        57.42           6.76     +49.38      +0.37
# Centroid 2D          4.56         2.65        65.95         126.55     +61.39    +123.90
# SQUID+s20           25.22        88.06        68.83          64.00     +43.61     -24.07
# SQUID+s10           43.91        99.76        70.86          74.89     +26.95     -24.87
# SQUID+s4            64.06        87.71        88.31          93.78     +24.25      +6.07
# SQUID               70.74        72.42        89.52          93.93     +18.78     +21.51
# SQUID+s2            69.59        72.47        91.16          90.05     +21.57     +17.58
# SQUID+s0            77.93        52.58        88.44          81.21     +10.51     +28.63
# Zero               101.20         7.50       101.20           7.50      -0.00      +0.00
# ====================================================================================================
#
# ====================================================================================================
# TABLE 4: PER-SITE BREAKDOWN (top 5 methods by avg MAE)
# ====================================================================================================
#   Site        X        Y        Z  Cal?   CC 2D Sp      CC 2D   CC 2D+JP      CC 1D   Centroid
# ----------------------------------------------------------------------------------------------------
#      0   29.970   22.900   3.3797   Yes       0.64       0.71       0.70       0.52      25.51
#      1   24.230   24.210   3.4000             5.94       6.07       6.07       6.61      28.63
#      2   25.000   27.360   3.3906             5.76       6.10       6.10       9.78      40.34
#      3   47.790   27.360   3.3422             2.09       2.18       2.18       3.22      30.86
# ====================================================================================================


# All-sites calibration
# uv run python "${BENCHMARK_SCRIPT}" --microscope "squid-royal-pearl" \
#     --cal-range 400 --cal-steps 29 \
#     --test-range 400 --test-steps 81 \
#     --positions "${POSITIONS_FILE}" \
#     --no-display --laf-exposure 5.0 --cal-all-sites

# RESULTS (all-sites calibration):
# ========================================================================================================================
# TABLE 2: AVERAGE RESULTS ACROSS 4 POSITIONS
# ========================================================================================================================
# Tracker         Avg MAE   |z|<=15   |z|<=5   MaxErr  Method
# ------------------------------------------------------------------------------------------------------------------------
# CC 2D Spline       2.96     2.60     2.39      6.0  2D CC + spline interp
# CC 2D              3.08     2.61     2.44      7.8  2D cross-corr full img
# CC 2D+JPEG         3.08     2.61     2.44      7.8  2D CC w/ JPEG compression
# CC 1D              3.65     2.65     2.52      9.6  1D cross-corr of profiles
# CNN                6.18     4.70     1.72     32.9  CNN on 64x64 resized
# Centroid          26.40     9.80     9.74     95.4  intensity centroid tracking
# Peak 1D           31.31    17.47    17.23     86.2  track peak in 1D profile
# Gauss 1D          31.50    17.73    17.80     85.1  gaussian fit to 1D profile
# SQUID+s50         38.62    26.77    26.82    136.4  SQUID + gaussian s=50
# Centroid 2D       50.21    57.66    46.61    139.2  centroid + brightness 2D lookup
# SQUID+s10         69.73    90.78    87.00    192.2  SQUID + gaussian s=10
# SQUID+s4          84.00    91.45    74.63    233.0  SQUID + gaussian s=4
# SQUID+s0          86.44    86.26    68.97    272.4  SQUID, no smoothing
# SQUID             88.57    86.16    72.87    253.9  real SQUID: blur s=1 + peak
# SQUID+s2          88.96    86.64    70.29    252.0  SQUID + gaussian s=2
# SQUID+s20         93.11    83.54    84.36    235.9  SQUID + gaussian s=20
# Zero             101.20     7.50     2.50    200.0  always predicts 0
# ========================================================================================================================
#
# TABLE 3: SKIPPED (all sites used for calibration)
#
# ====================================================================================================
# TABLE 4: PER-SITE BREAKDOWN (top 5 methods by avg MAE)
# ====================================================================================================
#   Site        X        Y        Z  Cal?   CC 2D Sp      CC 2D   CC 2D+JP      CC 1D        CNN
# ----------------------------------------------------------------------------------------------------
#      0   29.970   22.900   3.3797   Yes       3.47       3.55       3.55       3.85       6.91
#      1   24.230   24.210   3.4000   Yes       2.75       2.92       2.92       2.66       7.14
#      2   25.000   27.360   3.3906   Yes       2.67       2.99       2.99       4.83       5.67
#      3   47.790   27.360   3.3422   Yes       2.94       2.86       2.85       3.26       5.01
# ====================================================================================================
