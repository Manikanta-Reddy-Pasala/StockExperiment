# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 2968.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 6 |
| PENDING | 12 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 4 / 5 / 0
- **Avg / median % per leg:** 3.03% / -1.79%
- **Sum % (uncompounded):** 27.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 2 | 2 | 0 | 3.39% | 13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 2 | 2 | 0 | 3.39% | 13.6% |
| SELL (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.74% | 13.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.6% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 2 | 2 | 0 | 4.09% | 16.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.6% |
| retest2 (combined) | 8 | 4 | 50.0% | 4 | 4 | 0 | 3.74% | 29.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2584.25 | 2685.44 | 2685.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2575.00 | 2684.34 | 2684.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 2622.10 | 2611.71 | 2641.83 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-27 11:15:00 | 2598.10 | 2612.25 | 2639.94 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-27 12:15:00 | 2627.85 | 2612.41 | 2639.88 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 2601.00 | 2612.72 | 2639.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:15:00 | 2589.60 | 2612.49 | 2639.12 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2658.10 | 2611.11 | 2636.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 2658.10 | 2611.11 | 2636.83 | SL hit (close>ema400) qty=1.00 sl=2636.83 alert=retest1 |
| Cross detected — sustain check pending | 2024-12-18 10:15:00 | 2590.00 | 2647.10 | 2650.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:15:00 | 2594.15 | 2646.57 | 2650.39 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-18 13:15:00 | 2596.55 | 2645.61 | 2649.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:15:00 | 2592.50 | 2645.08 | 2649.58 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-01-13 09:15:00 | 2334.73 | 2522.43 | 2571.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 2333.25 | 2522.43 | 2571.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 2626.95 | 2473.04 | 2472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 2641.20 | 2493.19 | 2482.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2509.00 | 2509.44 | 2491.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 11:15:00 | 2486.00 | 2509.29 | 2492.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 2486.00 | 2509.29 | 2492.03 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 2581.50 | 2509.55 | 2492.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 2582.30 | 2510.28 | 2493.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-02 11:15:00 | 2520.30 | 2661.22 | 2630.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 2523.50 | 2659.84 | 2630.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-06-24 10:15:00 | 2775.85 | 2666.85 | 2643.84 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-25 15:15:00 | 2840.53 | 2681.86 | 2652.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.62 | 2803.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 2719.50 | 2799.80 | 2801.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.47 | 2778.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 2769.60 | 2762.54 | 2778.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2769.60 | 2762.54 | 2778.37 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-09 13:15:00 | 2761.00 | 2762.53 | 2778.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 2748.40 | 2762.39 | 2778.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 12:15:00 | 2748.20 | 2762.28 | 2777.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 2746.40 | 2762.12 | 2777.53 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.60 | 2762.55 | 2777.22 | SL hit (close>static) qty=1.00 sl=2779.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.60 | 2762.55 | 2777.22 | SL hit (close>static) qty=1.00 sl=2779.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2788.15 | 2787.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.15 | 2790.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.80 | 2804.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.64 | 2804.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.64 | 2804.46 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.58 | 2796.66 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.78 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2858.10 | 2798.35 | 2797.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 2772.80 | 2801.63 | 2799.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 2773.60 | 2801.35 | 2799.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-02 13:15:00 | 2777.50 | 2853.14 | 2836.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:15:00 | 2782.30 | 2852.43 | 2836.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2687.80 | 2850.05 | 2835.37 | SL hit (close<static) qty=1.00 sl=2722.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2687.80 | 2850.05 | 2835.37 | SL hit (close<static) qty=1.00 sl=2722.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.70 | 2821.62 | 2821.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2628.50 | 2793.62 | 2806.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2696.20 | 2686.31 | 2728.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 2730.40 | 2686.75 | 2728.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-16 12:15:00 | 2699.90 | 2693.64 | 2728.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 13:15:00 | 2720.50 | 2693.91 | 2728.64 | ENTRY2 sustain failed after 60m |

### Cycle 8 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 2871.50 | 2748.71 | 2748.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.80 | 2750.08 | 2749.20 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-28 11:15:00 | 2589.60 | 2024-12-02 09:15:00 | 2658.10 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-12-18 11:15:00 | 2594.15 | 2025-01-13 09:15:00 | 2334.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-18 14:15:00 | 2592.50 | 2025-01-13 09:15:00 | 2333.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 2582.30 | 2025-06-24 10:15:00 | 2775.85 | TARGET_HIT | 1.00 | 7.50% |
| BUY | retest2 | 2025-06-02 12:15:00 | 2523.50 | 2025-06-25 15:15:00 | 2840.53 | TARGET_HIT | 1.00 | 12.56% |
| SELL | retest2 | 2025-12-09 14:15:00 | 2748.40 | 2025-12-11 13:15:00 | 2797.60 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-10 13:15:00 | 2746.40 | 2025-12-11 13:15:00 | 2797.60 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-02-02 15:15:00 | 2773.60 | 2026-03-04 09:15:00 | 2687.80 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-02 14:15:00 | 2782.30 | 2026-03-04 09:15:00 | 2687.80 | STOP_HIT | 1.00 | -3.40% |
