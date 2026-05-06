# DIVISLAB (DIVISLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 6702.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 12 |
| PENDING | 54 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 6 |
| ENTRY2 | 39 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 39
- **Target hits / Stop hits / Partials:** 1 / 44 / 4
- **Avg / median % per leg:** 0.62% / -1.39%
- **Sum % (uncompounded):** 30.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 10 | 21.7% | 1 | 41 | 4 | 0.78% | 35.7% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 1 | 4 | 4 | 12.00% | 108.0% |
| BUY @ 3rd Alert (retest2) | 37 | 3 | 8.1% | 0 | 37 | 0 | -1.95% | -72.3% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.83% | -5.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.20% | -1.2% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.15% | -4.3% |
| retest1 (combined) | 10 | 7 | 70.0% | 1 | 5 | 4 | 10.68% | 106.8% |
| retest2 (combined) | 39 | 3 | 7.7% | 0 | 39 | 0 | -1.96% | -76.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 3797.95 | 3619.60 | 3619.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 3816.70 | 3635.97 | 3627.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 11:15:00 | 3666.00 | 3666.16 | 3646.07 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 3648.90 | 3666.10 | 3647.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 3648.90 | 3666.10 | 3647.02 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-13 14:15:00 | 3661.60 | 3664.50 | 3646.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 3661.70 | 3664.48 | 3646.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 3631.00 | 3676.16 | 3655.97 | SL hit qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2023-12-22 09:15:00 | 3702.00 | 3672.69 | 3655.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 10:15:00 | 3721.15 | 3673.17 | 3655.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 3631.00 | 3834.84 | 3771.07 | SL hit qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-18 10:15:00 | 3659.05 | 3833.09 | 3770.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:15:00 | 3689.20 | 3831.66 | 3770.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 3631.00 | 3812.37 | 3764.30 | SL hit qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-24 09:15:00 | 3659.45 | 3802.93 | 3760.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:15:00 | 3658.90 | 3801.50 | 3760.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 3616.40 | 3799.66 | 3759.48 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-24 11:15:00 | 3631.00 | 3799.66 | 3759.48 | SL hit qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-31 14:15:00 | 3675.70 | 3745.54 | 3736.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 3663.00 | 3744.72 | 3735.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-01 12:15:00 | 3660.05 | 3741.18 | 3734.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-01 13:15:00 | 3648.85 | 3740.26 | 3733.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-02 09:15:00 | 3724.85 | 3738.45 | 3732.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:15:00 | 3716.00 | 3738.23 | 3732.77 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-09 11:15:00 | 3666.90 | 3728.46 | 3728.43 | ENTRY2 cross detected — sustain check pending (15m) |
| CROSSOVER_SKIP | 2024-02-09 12:15:00 | 3652.10 | 3727.70 | 3728.05 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Sustain check cancelled (price retraced) | 2024-02-09 13:15:00 | 3646.75 | 3726.90 | 3727.64 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-02-12 09:15:00 | 3771.60 | 3725.87 | 3727.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 10:15:00 | 3755.80 | 3726.17 | 3727.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-14 11:15:00 | 3663.15 | 3724.41 | 3726.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 12:15:00 | 3691.00 | 3724.07 | 3726.14 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 3683.80 | 3723.67 | 3725.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-02-14 14:15:00 | 3721.80 | 3723.65 | 3725.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 15:15:00 | 3719.65 | 3723.61 | 3725.88 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-20 12:15:00 | 3704.65 | 3723.95 | 3725.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 13:15:00 | 3707.40 | 3723.78 | 3725.72 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 3671.40 | 3721.60 | 3724.55 | SL hit qty=1.00 sl=3671.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 3671.40 | 3721.60 | 3724.55 | SL hit qty=1.00 sl=3671.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 3612.30 | 3717.34 | 3722.32 | SL hit qty=1.00 sl=3612.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 3612.30 | 3717.34 | 3722.32 | SL hit qty=1.00 sl=3612.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 3612.30 | 3717.34 | 3722.32 | SL hit qty=1.00 sl=3612.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 3612.30 | 3717.34 | 3722.32 | SL hit qty=1.00 sl=3612.30 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-03 13:15:00 | 3713.00 | 3538.15 | 3593.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-03 14:15:00 | 3687.40 | 3539.64 | 3593.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-04 09:15:00 | 3700.00 | 3542.67 | 3594.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 10:15:00 | 3724.40 | 3544.47 | 3595.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 3812.95 | 3658.14 | 3647.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 3771.10 | 3799.05 | 3733.83 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-05-10 09:15:00 | 3820.50 | 3798.88 | 3735.03 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-10 10:15:00 | 3789.00 | 3798.78 | 3735.30 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-13 10:15:00 | 3814.60 | 3798.26 | 3737.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 3903.25 | 3799.31 | 3738.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-06-05 10:15:00 | 4488.74 | 4077.49 | 3928.76 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Target hit — 30% from entry | 2024-08-28 13:15:00 | 5074.23 | 4778.21 | 4629.05 | Target hit (30%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5828.40 | 5939.02 | 5799.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-20 09:15:00 | 5865.00 | 5932.04 | 5800.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 5901.95 | 5931.74 | 5800.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-23 09:15:00 | 5894.00 | 5929.42 | 5803.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 5888.30 | 5929.01 | 5803.92 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-24 15:15:00 | 5785.65 | 5918.68 | 5805.91 | SL hit qty=1.00 sl=5785.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 15:15:00 | 5785.65 | 5918.68 | 5805.91 | SL hit qty=1.00 sl=5785.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-26 14:15:00 | 5885.65 | 5911.74 | 5805.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 5886.75 | 5911.49 | 5806.12 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 13:15:00 | 5874.95 | 5944.67 | 5853.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 5865.35 | 5943.88 | 5853.80 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 5844.10 | 5942.15 | 5853.83 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 5785.65 | 5933.86 | 5852.65 | SL hit qty=1.00 sl=5785.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 5785.65 | 5933.86 | 5852.65 | SL hit qty=1.00 sl=5785.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 5946.95 | 5912.30 | 5847.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 5947.85 | 5912.66 | 5848.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 5815.00 | 5913.39 | 5849.80 | SL hit qty=1.00 sl=5815.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-15 11:15:00 | 5907.00 | 5912.98 | 5850.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 5916.10 | 5913.01 | 5850.56 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 5815.00 | 5911.96 | 5851.26 | SL hit qty=1.00 sl=5815.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-16 10:15:00 | 5907.00 | 5911.91 | 5851.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 5922.10 | 5912.01 | 5851.89 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-16 13:15:00 | 5906.00 | 5911.77 | 5852.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 5913.20 | 5911.78 | 5852.67 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 5852.15 | 5910.68 | 5854.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-20 10:15:00 | 5937.00 | 5910.95 | 5855.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 5931.45 | 5911.15 | 5855.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 5815.00 | 5912.63 | 5859.02 | SL hit qty=1.00 sl=5815.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 5815.00 | 5912.63 | 5859.02 | SL hit qty=1.00 sl=5815.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 5840.05 | 5912.63 | 5859.02 | SL hit qty=1.00 sl=5840.05 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-30 13:15:00 | 5650.85 | 5819.20 | 5819.59 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 6163.70 | 5801.45 | 5810.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 6170.95 | 5805.13 | 5811.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-04 15:15:00 | 6096.25 | 5819.03 | 5818.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 6096.25 | 5819.03 | 5818.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 6176.25 | 5822.58 | 5820.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 5863.35 | 5886.07 | 5855.59 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-11 14:15:00 | 5949.80 | 5886.83 | 5856.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 15:15:00 | 5946.75 | 5887.43 | 5856.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 5836.50 | 5886.92 | 5856.63 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 5856.63 | 5886.92 | 5856.63 | SL hit qty=1.00 sl=5856.63 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-12 12:15:00 | 5970.35 | 5888.82 | 5858.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-12 13:15:00 | 5942.05 | 5889.35 | 5858.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-12 14:15:00 | 5974.50 | 5890.20 | 5859.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 5969.85 | 5890.99 | 5859.58 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-14 12:15:00 | 5805.00 | 5900.53 | 5866.26 | SL hit qty=1.00 sl=5805.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 10:15:00 | 5969.30 | 5896.93 | 5868.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:15:00 | 5966.75 | 5897.62 | 5868.87 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 5805.00 | 5899.64 | 5870.61 | SL hit qty=1.00 sl=5805.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-02-28 10:15:00 | 5436.05 | 5845.59 | 5845.95 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-04-22 09:15:00 | 5973.50 | 5675.41 | 5715.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:15:00 | 6012.00 | 5678.76 | 5717.25 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-24 09:15:00 | 6204.00 | 5712.31 | 5732.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 6234.00 | 5717.50 | 5734.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 6067.00 | 5751.52 | 5751.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 6067.00 | 5751.52 | 5751.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.52 | 5751.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.13 | 5758.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.76 | 5840.34 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 5948.50 | 5906.19 | 5840.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 12:15:00 | 5950.00 | 5906.62 | 5841.43 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 10:15:00 | 5990.50 | 5910.64 | 5845.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:15:00 | 5996.00 | 5911.49 | 5845.83 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 15:15:00 | 5950.00 | 5913.03 | 5847.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:15:00 | 6136.00 | 5915.25 | 5849.34 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-12 09:15:00 | 6842.50 | 6439.32 | 6231.85 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-03 09:15:00 | 6895.40 | 6586.93 | 6405.81 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 09:15:00 | 7056.40 | 6645.50 | 6454.68 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6582.00 | 6694.32 | 6568.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-29 11:15:00 | 6607.00 | 6690.29 | 6568.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 6661.50 | 6690.01 | 6569.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 6609.00 | 6684.74 | 6574.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 6610.50 | 6684.00 | 6574.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 6555.50 | 6680.17 | 6574.59 | SL hit qty=1.00 sl=6555.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 6555.50 | 6680.17 | 6574.59 | SL hit qty=1.00 sl=6555.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 13:15:00 | 6136.00 | 6614.92 | 6552.57 | SL hit qty=0.50 sl=6136.00 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 5996.00 | 6593.42 | 6542.95 | SL hit qty=0.50 sl=5996.00 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 5950.00 | 6535.83 | 6516.10 | SL hit qty=0.50 sl=5950.00 alert=retest1 |
| CROSSOVER_SKIP | 2025-08-11 14:15:00 | 5992.00 | 6496.40 | 6496.71 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-10-13 11:15:00 | 6595.00 | 6058.80 | 6144.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 6592.50 | 6064.11 | 6146.80 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-13 14:15:00 | 6555.50 | 6073.97 | 6150.93 | SL hit qty=1.00 sl=6555.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 6607.50 | 6145.75 | 6182.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-16 10:15:00 | 6569.50 | 6149.97 | 6184.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 6606.50 | 6154.51 | 6186.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 6602.00 | 6158.96 | 6188.43 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 6593.00 | 6217.59 | 6216.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.59 | 6216.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6229.09 | 6222.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.56 | 6400.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 6406.00 | 6490.89 | 6407.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6406.00 | 6490.89 | 6407.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-25 13:15:00 | 6438.50 | 6473.23 | 6404.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-25 14:15:00 | 6410.00 | 6472.60 | 6404.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 6474.00 | 6472.02 | 6405.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 6480.00 | 6472.10 | 6405.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 6402.00 | 6470.52 | 6413.28 | SL hit qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-02 10:15:00 | 6425.00 | 6470.07 | 6413.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 11:15:00 | 6409.50 | 6469.46 | 6413.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 6425.00 | 6465.02 | 6412.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 12:15:00 | 6415.50 | 6464.52 | 6413.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 13:15:00 | 6431.50 | 6464.20 | 6413.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 6460.50 | 6464.16 | 6413.33 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 6402.00 | 6463.46 | 6415.22 | SL hit qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 10:15:00 | 6430.00 | 6463.13 | 6415.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 6440.00 | 6462.90 | 6415.41 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 6402.00 | 6462.40 | 6416.33 | SL hit qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 6483.00 | 6438.15 | 6408.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 6452.50 | 6438.30 | 6408.69 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 6428.00 | 6438.04 | 6408.85 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 6402.00 | 6438.04 | 6408.85 | SL hit qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 15:15:00 | 6438.00 | 6438.04 | 6409.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 6456.50 | 6438.23 | 6409.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 6400.50 | 6437.22 | 6409.72 | SL hit qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 6565.50 | 6416.51 | 6402.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 6514.00 | 6417.48 | 6402.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 6400.50 | 6438.83 | 6416.67 | SL hit qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 6506.00 | 6417.41 | 6408.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 6543.50 | 6418.66 | 6409.60 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6400.50 | 6464.77 | 6435.47 | SL hit qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-12 13:15:00 | 6445.00 | 6463.17 | 6435.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 6490.50 | 6463.44 | 6435.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 6433.50 | 6463.11 | 6435.63 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 6400.50 | 6462.16 | 6435.43 | SL hit qty=1.00 sl=6400.50 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6410.41 | 6411.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6406.56 | 6410.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 6290.00 | 6244.55 | 6315.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 12:15:00 | 6224.50 | 6244.35 | 6315.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-03 13:15:00 | 6202.00 | 6243.93 | 6314.75 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6198.42 | 6276.47 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6276.47 | 6198.42 | 6276.47 | SL hit qty=1.00 sl=6276.47 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 6224.50 | 6204.90 | 6277.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 6196.00 | 6204.81 | 6276.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 6305.00 | 6204.78 | 6265.79 | SL hit qty=1.00 sl=6305.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-13 09:15:00 | 6169.00 | 6295.39 | 6297.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 6149.00 | 6293.93 | 6296.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-16 13:15:00 | 6305.00 | 6080.27 | 6150.88 | SL hit qty=1.00 sl=6305.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 6468.00 | 6201.43 | 6201.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 6519.00 | 6210.13 | 6205.72 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-13 15:15:00 | 3661.70 | 2023-12-20 14:15:00 | 3631.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-12-22 10:15:00 | 3721.15 | 2024-01-18 09:15:00 | 3631.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-01-18 11:15:00 | 3689.20 | 2024-01-23 11:15:00 | 3631.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-01-24 10:15:00 | 3658.90 | 2024-01-24 11:15:00 | 3631.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-01-31 15:15:00 | 3663.00 | 2024-02-21 13:15:00 | 3671.40 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-02-02 10:15:00 | 3716.00 | 2024-02-21 13:15:00 | 3671.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-02-12 10:15:00 | 3755.80 | 2024-02-22 11:15:00 | 3612.30 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-02-14 12:15:00 | 3691.00 | 2024-02-22 11:15:00 | 3612.30 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-02-14 15:15:00 | 3719.65 | 2024-02-22 11:15:00 | 3612.30 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-02-20 13:15:00 | 3707.40 | 2024-02-22 11:15:00 | 3612.30 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-04-04 10:15:00 | 3724.40 | 2024-04-18 09:15:00 | 3733.70 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest1 | 2024-05-13 11:15:00 | 3903.25 | 2024-06-05 10:15:00 | 4488.74 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-05-13 11:15:00 | 3903.25 | 2024-08-28 13:15:00 | 5074.23 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-12-20 10:15:00 | 5901.95 | 2024-12-24 15:15:00 | 5785.65 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-12-23 10:15:00 | 5888.30 | 2024-12-24 15:15:00 | 5785.65 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-12-26 15:15:00 | 5886.75 | 2025-01-10 09:15:00 | 5785.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-01-08 14:15:00 | 5865.35 | 2025-01-10 09:15:00 | 5785.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-01-14 13:15:00 | 5947.85 | 2025-01-15 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-01-15 12:15:00 | 5916.10 | 2025-01-16 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-01-16 11:15:00 | 5922.10 | 2025-01-21 14:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-01-16 14:15:00 | 5913.20 | 2025-01-21 14:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-01-20 11:15:00 | 5931.45 | 2025-01-21 14:15:00 | 5840.05 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-02-04 10:15:00 | 6170.95 | 2025-02-04 15:15:00 | 6096.25 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2025-02-11 15:15:00 | 5946.75 | 2025-02-12 09:15:00 | 5856.63 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-02-12 15:15:00 | 5969.85 | 2025-02-14 12:15:00 | 5805.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-02-20 11:15:00 | 5966.75 | 2025-02-21 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-04-22 10:15:00 | 6012.00 | 2025-04-25 11:15:00 | 6067.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-04-24 10:15:00 | 6234.00 | 2025-04-25 11:15:00 | 6067.00 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-06-12 09:15:00 | 6842.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.00 | 2025-07-03 09:15:00 | 6895.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-07-08 09:15:00 | 7056.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-08-01 09:15:00 | 6555.50 | STOP_HIT | 0.50 | 10.18% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.00 | 2025-08-01 09:15:00 | 6555.50 | STOP_HIT | 0.50 | 9.33% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-08-06 13:15:00 | 6136.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-07-29 12:15:00 | 6661.50 | 2025-08-07 10:15:00 | 5996.00 | STOP_HIT | 1.00 | -9.99% |
| BUY | retest2 | 2025-07-31 13:15:00 | 6610.50 | 2025-08-08 14:15:00 | 5950.00 | STOP_HIT | 1.00 | -9.99% |
| BUY | retest2 | 2025-10-13 12:15:00 | 6592.50 | 2025-10-13 14:15:00 | 6555.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-16 12:15:00 | 6602.00 | 2025-10-20 12:15:00 | 6593.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-11-26 10:15:00 | 6480.00 | 2025-12-02 09:15:00 | 6402.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-03 14:15:00 | 6460.50 | 2025-12-05 09:15:00 | 6402.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-05 11:15:00 | 6440.00 | 2025-12-08 09:15:00 | 6402.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-11 12:15:00 | 6452.50 | 2025-12-11 14:15:00 | 6402.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-12 09:15:00 | 6456.50 | 2025-12-15 09:15:00 | 6400.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-19 10:15:00 | 6514.00 | 2025-12-29 09:15:00 | 6400.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-06 10:15:00 | 6543.50 | 2026-01-12 09:15:00 | 6400.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-01-12 14:15:00 | 6490.50 | 2026-01-13 11:15:00 | 6400.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest1 | 2026-02-03 13:15:00 | 6202.00 | 2026-02-11 11:15:00 | 6276.47 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-12 12:15:00 | 6196.00 | 2026-02-19 09:15:00 | 6305.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-03-13 10:15:00 | 6149.00 | 2026-04-16 13:15:00 | 6305.00 | STOP_HIT | 1.00 | -2.54% |
