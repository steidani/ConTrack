# ConTrack
Author: Daniel Steinfeld, daniel.steinfeld@alumni.ethz.ch

Contour Tracking (ConTrack) to detect and track circulation anomalies.

__!! Under Construction !!__

## Anomaly index following Schwierz et al. 2004
The ANOM index follows Schwierz et al. (2004), but instead of vertically-integrated PV, by using 500hPa geopotential height Z500 to track anticyclonic anomalies.
The following steps are carried out in its calculation:
1. Daily Z500 climatology is obtaines by taking a 31-day running mean over the baseline period (1981 - 2010)
2. A daily (or subdaily for higher temporal resolution) anomaly is calculated by taking the difference between the original Z500 data and the climatology from step 1 for the corresponding day. A anomaly threshold is obtained by calculating the 90th percentile of these differences and smoothing it with a 60-day rolling mean.


