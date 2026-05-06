# BAJAJ-AUTO (BAJAJ-AUTO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4997 bars)
- **Last close:** 10319.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 24 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 1 |
| ENTRY2 | 17 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 2 / 11 / 3
- **Avg / median % per leg:** 4.80% / -0.01%
- **Sum % (uncompounded):** 76.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 2 | 11 | 3 | 4.80% | 76.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 22.50% | 45.0% |
| BUY @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 1 | 11 | 2 | 2.27% | 31.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 22.50% | 45.0% |
| retest2 (combined) | 14 | 5 | 35.7% | 1 | 11 | 2 | 2.27% | 31.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 5091.95 | 4732.65 | 4732.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 12:15:00 | 5100.50 | 4736.31 | 4734.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 15:15:00 | 4906.05 | 4911.66 | 4842.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-10-05 09:15:00 | 4954.95 | 4912.09 | 4842.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:15:00 | 4962.05 | 4912.58 | 4843.16 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-11-21 09:15:00 | 5706.36 | 5349.58 | 5186.03 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Target hit — 30% from entry | 2023-12-18 09:15:00 | 6450.66 | 5906.81 | 5613.43 | Target hit (30%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 9416.00 | 9500.12 | 9294.00 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-18 12:15:00 | 9496.15 | 9500.08 | 9295.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 9577.00 | 9500.85 | 9296.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-19 13:15:00 | 9448.70 | 9502.87 | 9304.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-19 14:15:00 | 9390.65 | 9501.75 | 9304.93 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 9292.95 | 9499.09 | 9305.55 | SL hit qty=1.00 sl=9292.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-23 10:15:00 | 9470.10 | 9492.10 | 9309.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 9471.60 | 9491.90 | 9310.37 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 9292.95 | 9490.05 | 9310.35 | SL hit qty=1.00 sl=9292.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 9429.95 | 9459.25 | 9309.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 9465.30 | 9459.31 | 9310.53 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-08-30 14:15:00 | 10885.09 | 9968.21 | 9700.05 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-09-23 14:15:00 | 12304.89 | 11037.10 | 10466.61 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-11-04 12:15:00 | 9400.80 | 10773.45 | 10779.02 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2024-11-04 13:15:00 | 9505.85 | 10760.83 | 10772.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 9517.00 | 10748.46 | 10766.40 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 9525.55 | 10736.29 | 10760.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-05 09:15:00 | 9645.00 | 10725.43 | 10754.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 9776.70 | 10715.99 | 10749.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 9513.00 | 10429.08 | 10585.93 | SL hit qty=1.00 sl=9513.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-18 09:15:00 | 9559.55 | 10305.45 | 10511.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-18 10:15:00 | 9530.85 | 10297.74 | 10506.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-18 11:15:00 | 9567.70 | 10290.48 | 10501.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 12:15:00 | 9574.25 | 10283.35 | 10497.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-18 14:15:00 | 9513.00 | 10268.42 | 10487.50 | SL hit qty=1.00 sl=9513.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-19 09:15:00 | 9580.00 | 10254.16 | 10478.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 10:15:00 | 9615.00 | 10247.80 | 10473.85 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-19 15:15:00 | 9545.70 | 10216.91 | 10452.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2024-11-19 15:15:00 | 9513.00 | 10216.91 | 10452.64 | SL hit qty=1.00 sl=9513.00 alert=retest2 |
| Sustain check cancelled (price retraced) | 2024-11-21 09:15:00 | 9509.30 | 10209.87 | 10447.94 | ENTRY2 sustain failed after 2520m |
| Cross detected — sustain check pending | 2024-11-22 13:15:00 | 9586.15 | 10137.60 | 10398.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-22 14:15:00 | 9497.95 | 10131.24 | 10393.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-25 09:15:00 | 9567.00 | 10119.22 | 10384.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 9546.60 | 10113.53 | 10380.74 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2024-11-25 11:15:00 | 9546.45 | 10107.88 | 10376.58 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-07-17 13:15:00 | 8321.00 | 8363.90 | 8364.00 | slope filter: EMA200 not falling 0.50% over 350 bars |
| CROSSOVER_SKIP | 2025-07-21 13:15:00 | 8405.50 | 8364.19 | 8364.11 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-07-22 14:15:00 | 8303.00 | 8363.81 | 8363.94 | slope filter: EMA200 not falling 0.50% over 350 bars |
| CROSSOVER_SKIP | 2025-07-23 13:15:00 | 8394.00 | 8364.29 | 8364.17 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-07-24 14:15:00 | 8285.50 | 8363.45 | 8363.76 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8827.00 | 8322.51 | 8321.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.45 | 8403.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.58 | 8716.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 8789.00 | 8831.02 | 8719.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 8794.00 | 8830.66 | 8720.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 8794.00 | 8829.57 | 8728.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 8791.00 | 8829.19 | 8728.83 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8668.00 | 8933.14 | 8850.75 | SL hit qty=1.00 sl=8668.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8668.00 | 8933.14 | 8850.75 | SL hit qty=1.00 sl=8668.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 8866.00 | 8904.44 | 8842.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 8900.00 | 8904.40 | 8842.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 10:15:00 | 8795.50 | 8978.18 | 8932.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 8800.00 | 8976.41 | 8932.18 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 8816.50 | 8974.82 | 8931.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 8884.50 | 8969.28 | 8929.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 8943.00 | 8969.02 | 8929.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-25 10:15:00 | 10120.00 | 9648.48 | 9468.33 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 8800.00 | 9570.49 | 9497.22 | SL hit qty=0.50 sl=8800.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 8799.00 | 9570.49 | 9497.22 | SL hit qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 8925.00 | 9564.07 | 9494.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 8950.00 | 9557.96 | 9491.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2026-03-23 11:15:00 | 8799.00 | 9414.89 | 9425.00 | SL hit qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 8870.00 | 9385.38 | 9409.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-24 10:15:00 | 8820.00 | 9379.75 | 9406.86 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-24 11:15:00 | 8873.50 | 9374.72 | 9404.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 12:15:00 | 8980.00 | 9370.79 | 9402.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 8799.00 | 9305.73 | 9365.51 | SL hit qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-30 11:15:00 | 8852.00 | 9296.30 | 9360.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-30 12:15:00 | 8826.00 | 9291.62 | 9357.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 9008.50 | 9274.06 | 9347.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 8957.00 | 9270.90 | 9345.39 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2026-04-01 11:15:00 | 8950.00 | 9267.71 | 9343.41 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 3 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 9769.50 | 9371.94 | 9371.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 9817.00 | 9399.20 | 9385.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9441.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9757.00 | 9493.48 | 9442.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-05 10:15:00 | 4962.05 | 2023-11-21 09:15:00 | 5706.36 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2023-10-05 10:15:00 | 4962.05 | 2023-12-18 09:15:00 | 6450.66 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-07-18 13:15:00 | 9577.00 | 2024-07-22 09:15:00 | 9292.95 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-07-23 11:15:00 | 9471.60 | 2024-07-23 12:15:00 | 9292.95 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-07-26 10:15:00 | 9465.30 | 2024-08-30 14:15:00 | 10885.09 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-26 10:15:00 | 9465.30 | 2024-09-23 14:15:00 | 12304.89 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-11-04 14:15:00 | 9517.00 | 2024-11-13 09:15:00 | 9513.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-11-05 10:15:00 | 9776.70 | 2024-11-18 14:15:00 | 9513.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-11-18 12:15:00 | 9574.25 | 2024-11-19 15:15:00 | 9513.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-11-19 10:15:00 | 9615.00 | 2025-11-07 09:15:00 | 8668.00 | STOP_HIT | 1.00 | -9.85% |
| BUY | retest2 | 2024-11-25 10:15:00 | 9546.60 | 2025-11-07 09:15:00 | 8668.00 | STOP_HIT | 1.00 | -9.20% |
| BUY | retest2 | 2025-10-06 15:15:00 | 8794.00 | 2026-02-25 10:15:00 | 10120.00 | PARTIAL | 0.50 | 15.08% |
| BUY | retest2 | 2025-10-06 15:15:00 | 8794.00 | 2026-03-16 10:15:00 | 8800.00 | STOP_HIT | 0.50 | 0.07% |
| BUY | retest2 | 2025-10-09 12:15:00 | 8791.00 | 2026-03-16 10:15:00 | 8799.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-03-23 11:15:00 | 8799.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-18 11:15:00 | 8800.00 | 2026-03-30 09:15:00 | 8799.00 | STOP_HIT | 1.00 | -0.01% |
