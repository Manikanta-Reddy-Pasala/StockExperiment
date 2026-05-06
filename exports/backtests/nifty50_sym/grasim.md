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
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 6 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 8.02% / 12.32%
- **Sum % (uncompounded):** 48.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 8.02% | 48.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 8.02% | 48.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 8.02% | 48.1% |

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

### Cycle 5 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.70 | 2821.62 | 2821.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2628.50 | 2793.62 | 2806.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2696.20 | 2686.31 | 2728.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 2730.40 | 2686.75 | 2728.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-16 12:15:00 | 2699.90 | 2693.64 | 2728.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 13:15:00 | 2720.50 | 2693.91 | 2728.64 | ENTRY2 sustain failed after 60m |

### Cycle 6 — BUY (started 2026-05-05 15:15:00)

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
