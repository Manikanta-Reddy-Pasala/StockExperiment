# GRASIM (GRASIM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2914.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 9 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 3
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** 5.55% / 4.11%
- **Sum % (uncompounded):** 61.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 0 | 9 | 2 | 5.55% | 61.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 8 | 72.7% | 0 | 9 | 2 | 5.55% | 61.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 8 | 72.7% | 0 | 9 | 2 | 5.55% | 61.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 2626.95 | 2473.04 | 2472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 2641.20 | 2493.19 | 2482.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2509.00 | 2509.44 | 2491.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 11:15:00 | 2486.00 | 2509.29 | 2492.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 2486.00 | 2509.29 | 2492.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 2581.50 | 2509.55 | 2492.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 2582.30 | 2510.28 | 2493.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 2520.30 | 2661.22 | 2630.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 2523.50 | 2659.84 | 2630.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-22 10:15:00 | 2902.02 | 2810.08 | 2782.42 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-29 09:15:00 | 2969.65 | 2831.19 | 2807.91 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-11-21 10:15:00 | 2738.70 | 2803.62 | 2803.69 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 2834.30 | 2788.15 | 2787.95 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 2834.30 | 2788.15 | 2787.95 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2788.15 | 2787.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.15 | 2790.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.80 | 2804.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.64 | 2804.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.64 | 2804.46 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.58 | 2796.66 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.78 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2858.10 | 2798.35 | 2797.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 2772.80 | 2801.63 | 2799.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 2773.60 | 2801.35 | 2799.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-02 13:15:00 | 2777.50 | 2853.14 | 2836.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:15:00 | 2782.30 | 2852.43 | 2836.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2722.00 | 2850.05 | 2835.37 | SL hit qty=1.00 sl=2722.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2722.00 | 2850.05 | 2835.37 | SL hit qty=1.00 sl=2722.00 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-09 09:15:00 | 2661.70 | 2821.62 | 2821.82 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2026-04-08 10:15:00 | 2784.20 | 2672.36 | 2726.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 11:15:00 | 2775.10 | 2673.38 | 2726.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2722.00 | 2686.31 | 2728.28 | SL hit qty=1.00 sl=2722.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-21 09:15:00 | 2776.60 | 2698.43 | 2728.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 2780.00 | 2699.24 | 2729.03 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 2730.90 | 2710.76 | 2732.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 2757.70 | 2711.47 | 2732.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 2752.80 | 2711.88 | 2732.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-27 09:15:00 | 2763.20 | 2713.87 | 2733.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:15:00 | 2785.00 | 2714.57 | 2733.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 2760.60 | 2730.45 | 2739.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 2758.20 | 2730.73 | 2739.98 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.80 | 2750.08 | 2749.20 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-08 10:15:00 | 2582.30 | 2025-09-22 10:15:00 | 2902.02 | PARTIAL | 0.50 | 12.38% |
| BUY | retest2 | 2025-06-02 12:15:00 | 2523.50 | 2025-10-29 09:15:00 | 2969.65 | PARTIAL | 0.50 | 17.68% |
| BUY | retest2 | 2025-04-08 10:15:00 | 2582.30 | 2025-12-24 12:15:00 | 2834.30 | STOP_HIT | 0.50 | 9.76% |
| BUY | retest2 | 2025-06-02 12:15:00 | 2523.50 | 2025-12-24 12:15:00 | 2834.30 | STOP_HIT | 0.50 | 12.32% |
| BUY | retest2 | 2026-02-02 15:15:00 | 2773.60 | 2026-03-04 09:15:00 | 2722.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-02 14:15:00 | 2782.30 | 2026-03-04 09:15:00 | 2722.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-04-08 11:15:00 | 2775.10 | 2026-04-13 09:15:00 | 2722.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-21 10:15:00 | 2780.00 | 2026-05-05 15:15:00 | 2871.50 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2026-04-24 10:15:00 | 2752.80 | 2026-05-05 15:15:00 | 2871.50 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2026-04-27 10:15:00 | 2785.00 | 2026-05-05 15:15:00 | 2871.50 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2026-04-30 11:15:00 | 2758.20 | 2026-05-05 15:15:00 | 2871.50 | STOP_HIT | 1.00 | 4.11% |
