# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 6710.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 16 |
| PENDING | 52 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 37 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 41
- **Target hits / Stop hits / Partials:** 4 / 41 / 4
- **Avg / median % per leg:** -0.72% / -2.16%
- **Sum % (uncompounded):** -35.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 8 | 22.9% | 4 | 27 | 4 | 0.21% | 7.3% |
| BUY @ 2nd Alert (retest1) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.46% | 58.1% |
| BUY @ 3rd Alert (retest2) | 26 | 0 | 0.0% | 0 | 26 | 0 | -1.95% | -50.8% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -3.05% | -42.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.30% | -6.9% |
| SELL @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.25% | -35.8% |
| retest1 (combined) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.27% | 51.3% |
| retest2 (combined) | 37 | 0 | 0.0% | 0 | 37 | 0 | -2.34% | -86.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 3463.25 | 3679.43 | 3679.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 3433.65 | 3676.99 | 3678.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 3542.50 | 3535.98 | 3590.28 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-11-13 11:15:00 | 3496.50 | 3535.29 | 3589.38 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 12:15:00 | 3495.50 | 3534.89 | 3588.92 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 3597.00 | 3535.33 | 3584.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 3597.00 | 3535.33 | 3584.45 | SL hit (close>ema400) qty=1.00 sl=3584.45 alert=retest1 |

### Cycle 2 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 3797.95 | 3619.57 | 3619.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 3816.70 | 3635.95 | 3627.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 11:15:00 | 3666.00 | 3666.14 | 3645.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 3648.90 | 3666.08 | 3646.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 3648.90 | 3666.08 | 3646.77 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-13 14:15:00 | 3661.60 | 3664.49 | 3646.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 3661.70 | 3664.46 | 3646.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 3619.55 | 3676.15 | 3655.76 | SL hit (close<static) qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2023-12-22 09:15:00 | 3702.00 | 3672.68 | 3654.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 10:15:00 | 3721.15 | 3673.16 | 3655.20 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 3626.15 | 3834.84 | 3770.97 | SL hit (close<static) qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-18 10:15:00 | 3659.05 | 3833.09 | 3770.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:15:00 | 3689.20 | 3831.66 | 3770.00 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 3603.35 | 3810.28 | 3763.40 | SL hit (close<static) qty=1.00 sl=3631.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-24 09:15:00 | 3659.45 | 3802.93 | 3760.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:15:00 | 3658.90 | 3801.50 | 3760.11 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-24 11:15:00 | 3616.40 | 3799.66 | 3759.39 | SL hit (close<static) qty=1.00 sl=3631.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 3728.75 | 3734.18 | 3730.97 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-06 13:15:00 | 3740.55 | 3734.16 | 3731.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-06 14:15:00 | 3724.70 | 3734.07 | 3730.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-07 09:15:00 | 3741.10 | 3734.09 | 3731.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 3739.20 | 3734.14 | 3731.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-08 10:15:00 | 3672.15 | 3732.81 | 3730.49 | SL hit (close<static) qty=1.00 sl=3700.90 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 3652.10 | 3727.70 | 3727.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 13:15:00 | 3646.75 | 3726.89 | 3727.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 3771.60 | 3725.87 | 3727.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 3771.60 | 3725.87 | 3727.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 3771.60 | 3725.87 | 3727.05 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-02-14 09:15:00 | 3641.65 | 3725.77 | 3726.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 10:15:00 | 3650.10 | 3725.02 | 3726.58 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-21 14:15:00 | 3655.45 | 3720.94 | 3724.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 15:15:00 | 3641.00 | 3720.15 | 3723.74 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-23 13:15:00 | 3656.80 | 3711.14 | 3718.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 14:15:00 | 3648.15 | 3710.51 | 3718.52 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 3733.70 | 3634.90 | 3634.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 3812.95 | 3658.14 | 3646.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 3771.10 | 3799.05 | 3733.82 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-05-10 09:15:00 | 3820.50 | 3798.88 | 3735.02 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-10 10:15:00 | 3789.00 | 3798.78 | 3735.29 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-13 10:15:00 | 3814.60 | 3798.26 | 3737.21 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 3903.25 | 3799.31 | 3738.04 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:15:00 | 4098.41 | 3859.31 | 3784.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-05-27 09:15:00 | 4293.58 | 3891.31 | 3804.99 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5828.40 | 5939.02 | 5799.03 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-20 09:15:00 | 5865.00 | 5932.04 | 5800.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 5901.95 | 5931.74 | 5800.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-23 09:15:00 | 5894.00 | 5929.42 | 5803.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 5888.30 | 5929.01 | 5803.92 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-24 15:15:00 | 5782.35 | 5918.68 | 5805.91 | SL hit (close<static) qty=1.00 sl=5785.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 15:15:00 | 5782.35 | 5918.68 | 5805.91 | SL hit (close<static) qty=1.00 sl=5785.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-26 14:15:00 | 5885.65 | 5911.74 | 5805.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 5886.75 | 5911.49 | 5806.12 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 13:15:00 | 5874.95 | 5944.67 | 5853.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:15:00 | 5865.35 | 5943.88 | 5853.80 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 5844.10 | 5942.15 | 5853.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 5743.25 | 5933.86 | 5852.65 | SL hit (close<static) qty=1.00 sl=5785.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 5743.25 | 5933.86 | 5852.65 | SL hit (close<static) qty=1.00 sl=5785.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 5946.95 | 5912.30 | 5847.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 5947.85 | 5912.66 | 5848.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-15 11:15:00 | 5907.00 | 5912.98 | 5850.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 5916.10 | 5913.01 | 5850.56 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-16 10:15:00 | 5907.00 | 5911.91 | 5851.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 5922.10 | 5912.01 | 5851.89 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-16 13:15:00 | 5906.00 | 5911.77 | 5852.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 5913.20 | 5911.78 | 5852.67 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 5852.15 | 5910.68 | 5854.72 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-01-20 10:15:00 | 5937.00 | 5910.95 | 5855.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 5931.45 | 5911.15 | 5855.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 5819.55 | 5912.63 | 5859.02 | SL hit (close<static) qty=1.00 sl=5840.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 5776.50 | 5910.32 | 5858.40 | SL hit (close<static) qty=1.00 sl=5815.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 5776.50 | 5910.32 | 5858.40 | SL hit (close<static) qty=1.00 sl=5815.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 5776.50 | 5910.32 | 5858.40 | SL hit (close<static) qty=1.00 sl=5815.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 5776.50 | 5910.32 | 5858.40 | SL hit (close<static) qty=1.00 sl=5815.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 5650.85 | 5819.20 | 5819.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 5610.45 | 5815.25 | 5817.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 5805.15 | 5795.72 | 5807.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 5805.15 | 5795.72 | 5807.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 5805.15 | 5795.72 | 5807.46 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 6096.25 | 5819.03 | 5818.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 6176.25 | 5822.58 | 5820.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 5863.35 | 5886.07 | 5855.59 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-02-11 14:15:00 | 5949.80 | 5886.83 | 5856.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 15:15:00 | 5946.75 | 5887.43 | 5856.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 5836.50 | 5886.92 | 5856.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 5836.50 | 5886.92 | 5856.63 | SL hit (close<ema400) qty=1.00 sl=5856.63 alert=retest1 |
| Cross detected — sustain check pending | 2025-02-12 12:15:00 | 5970.35 | 5888.82 | 5858.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-12 13:15:00 | 5942.05 | 5889.35 | 5858.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-12 14:15:00 | 5974.50 | 5890.20 | 5859.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 5969.85 | 5890.99 | 5859.58 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-14 13:15:00 | 5804.15 | 5899.57 | 5865.95 | SL hit (close<static) qty=1.00 sl=5805.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 10:15:00 | 5969.30 | 5896.93 | 5868.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:15:00 | 5966.75 | 5897.62 | 5868.87 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 5788.05 | 5899.64 | 5870.61 | SL hit (close<static) qty=1.00 sl=5805.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 5436.05 | 5845.59 | 5845.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5373.15 | 5840.89 | 5843.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 5737.25 | 5709.48 | 5766.27 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-13 14:15:00 | 5618.60 | 5707.72 | 5763.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 15:15:00 | 5620.45 | 5706.85 | 5763.27 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 5767.90 | 5705.24 | 5758.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 5767.90 | 5705.24 | 5758.86 | SL hit (close>ema400) qty=1.00 sl=5758.86 alert=retest1 |
| Cross detected — sustain check pending | 2025-04-01 10:15:00 | 5655.30 | 5755.26 | 5774.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 5599.10 | 5753.71 | 5773.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 5825.80 | 5737.90 | 5763.90 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 5421.40 | 5736.48 | 5762.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 5515.50 | 5734.28 | 5761.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 5796.50 | 5650.84 | 5708.79 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-17 12:15:00 | 5693.00 | 5659.15 | 5710.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 13:15:00 | 5677.50 | 5659.34 | 5710.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 5822.50 | 5660.81 | 5710.05 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.52 | 5751.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.13 | 5758.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.76 | 5840.34 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 5948.50 | 5906.19 | 5840.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 12:15:00 | 5950.00 | 5906.62 | 5841.43 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 10:15:00 | 5990.50 | 5910.64 | 5845.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:15:00 | 5996.00 | 5911.49 | 5845.83 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 15:15:00 | 5950.00 | 5913.03 | 5847.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:15:00 | 6136.00 | 5915.25 | 5849.34 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 13:15:00 | 6247.50 | 5939.61 | 5865.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:15:00 | 6295.80 | 5965.45 | 5882.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-19 09:15:00 | 6545.00 | 5989.18 | 5897.11 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-05-19 09:15:00 | 6595.60 | 5989.18 | 5897.11 | Target hit (10%) qty=0.50 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:15:00 | 6442.80 | 5989.18 | 5897.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-26 10:15:00 | 6749.60 | 6157.98 | 6004.35 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6582.00 | 6694.32 | 6568.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-29 11:15:00 | 6607.00 | 6690.29 | 6568.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 6661.50 | 6690.01 | 6569.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 6609.00 | 6684.74 | 6574.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 6610.50 | 6684.00 | 6574.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 6468.00 | 6680.17 | 6574.59 | SL hit (close<static) qty=1.00 sl=6555.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 6468.00 | 6680.17 | 6574.59 | SL hit (close<static) qty=1.00 sl=6555.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 5992.00 | 6496.40 | 6496.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 5968.00 | 6486.17 | 6491.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6258.50 | 6145.76 | 6248.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 6258.50 | 6145.76 | 6248.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6258.50 | 6145.76 | 6248.34 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 6139.00 | 6149.72 | 6246.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 6144.50 | 6149.66 | 6246.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 14:15:00 | 6119.50 | 6008.85 | 6129.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 6120.50 | 6009.96 | 6129.14 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 6154.50 | 6014.75 | 6129.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-09 12:15:00 | 6163.00 | 6016.22 | 6129.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 6142.00 | 6017.47 | 6130.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 6131.00 | 6018.60 | 6130.00 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6385.50 | 6026.28 | 6131.67 | SL hit (close>static) qty=1.00 sl=6267.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6385.50 | 6026.28 | 6131.67 | SL hit (close>static) qty=1.00 sl=6267.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6385.50 | 6026.28 | 6131.67 | SL hit (close>static) qty=1.00 sl=6267.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.59 | 6216.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6229.09 | 6222.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.56 | 6400.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 6406.00 | 6490.89 | 6407.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6406.00 | 6490.89 | 6407.20 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-25 13:15:00 | 6438.50 | 6473.23 | 6404.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-25 14:15:00 | 6410.00 | 6472.60 | 6404.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 6474.00 | 6472.02 | 6405.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 6480.00 | 6472.10 | 6405.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 6374.50 | 6470.52 | 6413.28 | SL hit (close<static) qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-02 10:15:00 | 6425.00 | 6470.07 | 6413.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 11:15:00 | 6409.50 | 6469.46 | 6413.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 6425.00 | 6465.02 | 6412.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 12:15:00 | 6415.50 | 6464.52 | 6413.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 13:15:00 | 6431.50 | 6464.20 | 6413.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 6460.50 | 6464.16 | 6413.33 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-05 10:15:00 | 6430.00 | 6463.13 | 6415.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 6440.00 | 6462.90 | 6415.41 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 6401.00 | 6462.40 | 6416.33 | SL hit (close<static) qty=1.00 sl=6402.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 6401.00 | 6462.40 | 6416.33 | SL hit (close<static) qty=1.00 sl=6402.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-11 11:15:00 | 6483.00 | 6438.15 | 6408.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 6452.50 | 6438.30 | 6408.69 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 6428.00 | 6438.04 | 6408.85 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-11 15:15:00 | 6438.00 | 6438.04 | 6409.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 6456.50 | 6438.23 | 6409.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 6389.00 | 6437.22 | 6409.72 | SL hit (close<static) qty=1.00 sl=6402.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 6389.00 | 6437.22 | 6409.72 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 6565.50 | 6416.51 | 6402.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 6514.00 | 6417.48 | 6402.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 6390.50 | 6438.83 | 6416.67 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 6506.00 | 6417.41 | 6408.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 6543.50 | 6418.66 | 6409.60 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 6399.00 | 6464.77 | 6435.47 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-12 13:15:00 | 6445.00 | 6463.17 | 6435.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 6490.50 | 6463.44 | 6435.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 6433.50 | 6463.11 | 6435.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 6395.00 | 6462.16 | 6435.43 | SL hit (close<static) qty=1.00 sl=6400.50 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6410.41 | 6411.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6406.56 | 6410.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 6290.00 | 6244.55 | 6315.77 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-03 12:15:00 | 6224.50 | 6244.35 | 6315.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-03 13:15:00 | 6202.00 | 6243.93 | 6314.75 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6198.42 | 6276.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6198.42 | 6276.47 | SL hit (close>ema400) qty=1.00 sl=6276.47 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 6224.50 | 6204.90 | 6277.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 6196.00 | 6204.81 | 6276.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 6313.50 | 6204.78 | 6265.79 | SL hit (close>static) qty=1.00 sl=6305.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-13 09:15:00 | 6169.00 | 6295.39 | 6297.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 6149.00 | 6293.93 | 6296.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 6310.00 | 6114.91 | 6162.34 | SL hit (close>static) qty=1.00 sl=6305.00 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 6468.00 | 6201.43 | 6201.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 6519.00 | 6210.13 | 6205.72 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-11-13 12:15:00 | 3495.50 | 2023-11-17 09:15:00 | 3597.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2023-12-13 15:15:00 | 3661.70 | 2023-12-20 14:15:00 | 3619.55 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-12-22 10:15:00 | 3721.15 | 2024-01-18 09:15:00 | 3626.15 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-01-18 11:15:00 | 3689.20 | 2024-01-23 12:15:00 | 3603.35 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-01-24 10:15:00 | 3658.90 | 2024-01-24 11:15:00 | 3616.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-02-07 10:15:00 | 3739.20 | 2024-02-08 10:15:00 | 3672.15 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-02-14 10:15:00 | 3650.10 | 2024-04-18 09:15:00 | 3733.70 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-02-21 15:15:00 | 3641.00 | 2024-04-18 09:15:00 | 3733.70 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-02-23 14:15:00 | 3648.15 | 2024-04-18 09:15:00 | 3733.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2024-05-13 11:15:00 | 3903.25 | 2024-05-23 11:15:00 | 4098.41 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-05-13 11:15:00 | 3903.25 | 2024-05-27 09:15:00 | 4293.58 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-20 10:15:00 | 5901.95 | 2024-12-24 15:15:00 | 5782.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-23 10:15:00 | 5888.30 | 2024-12-24 15:15:00 | 5782.35 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-12-26 15:15:00 | 5886.75 | 2025-01-10 09:15:00 | 5743.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-01-08 14:15:00 | 5865.35 | 2025-01-10 09:15:00 | 5743.25 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-01-14 13:15:00 | 5947.85 | 2025-01-21 14:15:00 | 5819.55 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-01-15 12:15:00 | 5916.10 | 2025-01-22 09:15:00 | 5776.50 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-01-16 11:15:00 | 5922.10 | 2025-01-22 09:15:00 | 5776.50 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-01-16 14:15:00 | 5913.20 | 2025-01-22 09:15:00 | 5776.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-01-20 11:15:00 | 5931.45 | 2025-01-22 09:15:00 | 5776.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest1 | 2025-02-11 15:15:00 | 5946.75 | 2025-02-12 09:15:00 | 5836.50 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-02-12 15:15:00 | 5969.85 | 2025-02-14 13:15:00 | 5804.15 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-02-20 11:15:00 | 5966.75 | 2025-02-21 09:15:00 | 5788.05 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest1 | 2025-03-13 15:15:00 | 5620.45 | 2025-03-18 14:15:00 | 5767.90 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-04-01 11:15:00 | 5599.10 | 2025-04-03 09:15:00 | 5825.80 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-04-04 10:15:00 | 5515.50 | 2025-04-16 09:15:00 | 5796.50 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-04-17 13:15:00 | 5677.50 | 2025-04-21 09:15:00 | 5822.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-05-14 13:15:00 | 6247.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.00 | 2025-05-16 09:15:00 | 6295.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-09 12:15:00 | 5950.00 | 2025-05-19 09:15:00 | 6545.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-12 11:15:00 | 5996.00 | 2025-05-19 09:15:00 | 6595.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-05-19 09:15:00 | 6442.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-13 09:15:00 | 6136.00 | 2025-05-26 10:15:00 | 6749.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 12:15:00 | 6661.50 | 2025-08-01 09:15:00 | 6468.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-31 13:15:00 | 6610.50 | 2025-08-01 09:15:00 | 6468.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-09-22 10:15:00 | 6144.50 | 2025-10-10 11:15:00 | 6385.50 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-10-08 15:15:00 | 6120.50 | 2025-10-10 11:15:00 | 6385.50 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-10-09 14:15:00 | 6131.00 | 2025-10-10 11:15:00 | 6385.50 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2025-11-26 10:15:00 | 6480.00 | 2025-12-02 09:15:00 | 6374.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-03 14:15:00 | 6460.50 | 2025-12-08 09:15:00 | 6401.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-05 11:15:00 | 6440.00 | 2025-12-08 09:15:00 | 6401.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-11 12:15:00 | 6452.50 | 2025-12-15 09:15:00 | 6389.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 09:15:00 | 6456.50 | 2025-12-15 09:15:00 | 6389.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-19 10:15:00 | 6514.00 | 2025-12-29 09:15:00 | 6390.50 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-01-06 10:15:00 | 6543.50 | 2026-01-12 09:15:00 | 6399.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-01-12 14:15:00 | 6490.50 | 2026-01-13 11:15:00 | 6395.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest1 | 2026-02-03 13:15:00 | 6202.00 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-02-12 12:15:00 | 6196.00 | 2026-02-19 09:15:00 | 6313.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-03-13 10:15:00 | 6149.00 | 2026-04-22 09:15:00 | 6310.00 | STOP_HIT | 1.00 | -2.62% |
