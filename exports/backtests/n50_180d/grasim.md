# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 20
- **Target hits / Stop hits / Partials:** 0 / 20 / 0
- **Avg / median % per leg:** -1.06% / -0.91%
- **Sum % (uncompounded):** -21.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.89% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.89% | -7.1% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.17% | -14.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.17% | -14.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.06% | -21.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.65 | 2803.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 2723.00 | 2800.52 | 2802.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.38 | 2778.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 2763.90 | 2762.38 | 2778.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 2766.80 | 2762.46 | 2778.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2772.60 | 2762.24 | 2777.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 2757.20 | 2762.31 | 2777.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.21 | SL hit (close>static) qty=1.00 sl=2779.80 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2787.98 | 2787.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.03 | 2790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.82 | 2804.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 2801.00 | 2816.66 | 2804.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2799.30 | 2816.49 | 2804.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2799.30 | 2816.49 | 2804.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2789.70 | 2815.88 | 2804.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 2789.70 | 2815.88 | 2804.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2799.40 | 2815.18 | 2804.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:30:00 | 2810.00 | 2810.99 | 2802.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 2785.90 | 2810.38 | 2802.42 | SL hit (close<static) qty=1.00 sl=2789.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 2812.20 | 2808.80 | 2801.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 2809.40 | 2808.57 | 2801.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 2807.20 | 2808.65 | 2802.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 2802.70 | 2808.59 | 2802.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 2802.70 | 2808.59 | 2802.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2808.50 | 2808.59 | 2802.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 2810.40 | 2808.59 | 2802.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:30:00 | 2815.60 | 2808.51 | 2802.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2810.50 | 2808.51 | 2802.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 2795.00 | 2808.23 | 2802.13 | SL hit (close<static) qty=1.00 sl=2799.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 2795.00 | 2808.23 | 2802.13 | SL hit (close<static) qty=1.00 sl=2799.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 2795.00 | 2808.23 | 2802.13 | SL hit (close<static) qty=1.00 sl=2799.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2784.90 | 2808.00 | 2802.04 | SL hit (close<static) qty=1.00 sl=2789.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2784.90 | 2808.00 | 2802.04 | SL hit (close<static) qty=1.00 sl=2789.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 2784.90 | 2808.00 | 2802.04 | SL hit (close<static) qty=1.00 sl=2789.50 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.54 | 2796.63 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.82 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2857.90 | 2798.38 | 2797.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 2795.60 | 2805.26 | 2801.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2762.20 | 2804.83 | 2801.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2769.90 | 2804.83 | 2801.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2833.90 | 2857.65 | 2834.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 2833.90 | 2857.65 | 2834.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2835.00 | 2857.42 | 2834.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 2863.00 | 2857.42 | 2834.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 2810.90 | 2860.22 | 2839.09 | SL hit (close<static) qty=1.00 sl=2822.70 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2630.00 | 2793.23 | 2806.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 2734.40 | 2670.97 | 2725.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 2784.20 | 2672.10 | 2725.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 2784.20 | 2672.10 | 2725.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 2744.30 | 2678.10 | 2726.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 2731.80 | 2680.09 | 2727.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2765.00 | 2682.17 | 2727.58 | SL hit (close>static) qty=1.00 sl=2755.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 12:45:00 | 2733.30 | 2684.12 | 2727.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 13:30:00 | 2735.40 | 2684.65 | 2727.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 2732.60 | 2684.65 | 2727.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 2756.70 | 2685.98 | 2728.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 2756.70 | 2685.98 | 2728.18 | SL hit (close>static) qty=1.00 sl=2755.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 2756.70 | 2685.98 | 2728.18 | SL hit (close>static) qty=1.00 sl=2755.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 2756.70 | 2685.98 | 2728.18 | SL hit (close>static) qty=1.00 sl=2755.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 2696.70 | 2685.98 | 2728.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 2733.10 | 2691.58 | 2728.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:45:00 | 2735.00 | 2693.06 | 2728.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 2729.20 | 2693.26 | 2728.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2728.50 | 2694.43 | 2728.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 2728.50 | 2694.43 | 2728.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 2720.60 | 2694.69 | 2728.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:30:00 | 2718.20 | 2695.18 | 2728.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 14:00:00 | 2717.90 | 2695.40 | 2728.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 2706.10 | 2696.15 | 2727.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 2745.30 | 2696.96 | 2728.04 | SL hit (close>static) qty=1.00 sl=2736.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 2745.30 | 2696.96 | 2728.04 | SL hit (close>static) qty=1.00 sl=2736.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 2745.30 | 2696.96 | 2728.04 | SL hit (close>static) qty=1.00 sl=2736.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2760.40 | 2698.73 | 2728.47 | SL hit (close>static) qty=1.00 sl=2759.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2760.40 | 2698.73 | 2728.47 | SL hit (close>static) qty=1.00 sl=2759.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2760.40 | 2698.73 | 2728.47 | SL hit (close>static) qty=1.00 sl=2759.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2760.40 | 2698.73 | 2728.47 | SL hit (close>static) qty=1.00 sl=2759.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 2875.50 | 2748.38 | 2747.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.00 | 2750.88 | 2749.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-10 12:15:00 | 2757.20 | 2025-12-11 13:15:00 | 2797.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-12 14:30:00 | 2810.00 | 2026-01-13 11:15:00 | 2785.90 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-14 10:45:00 | 2812.20 | 2026-01-19 13:15:00 | 2795.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-16 09:45:00 | 2809.40 | 2026-01-19 13:15:00 | 2795.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-16 11:45:00 | 2807.20 | 2026-01-19 13:15:00 | 2795.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-16 14:45:00 | 2810.40 | 2026-01-19 14:15:00 | 2784.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-19 09:30:00 | 2815.60 | 2026-01-19 14:15:00 | 2784.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2810.50 | 2026-01-19 14:15:00 | 2784.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-23 09:15:00 | 2863.00 | 2026-02-27 10:15:00 | 2810.90 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-09 14:15:00 | 2731.80 | 2026-04-10 09:15:00 | 2765.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-04-10 12:45:00 | 2733.30 | 2026-04-10 15:15:00 | 2756.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-10 13:30:00 | 2735.40 | 2026-04-10 15:15:00 | 2756.70 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-04-10 14:15:00 | 2732.60 | 2026-04-10 15:15:00 | 2756.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2696.70 | 2026-04-20 11:15:00 | 2745.30 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-15 15:15:00 | 2733.10 | 2026-04-20 11:15:00 | 2745.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-16 10:45:00 | 2735.00 | 2026-04-20 11:15:00 | 2745.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-04-16 11:30:00 | 2729.20 | 2026-04-20 14:15:00 | 2760.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-17 12:30:00 | 2718.20 | 2026-04-20 14:15:00 | 2760.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-17 14:00:00 | 2717.90 | 2026-04-20 14:15:00 | 2760.40 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-20 09:30:00 | 2706.10 | 2026-04-20 14:15:00 | 2760.40 | STOP_HIT | 1.00 | -2.01% |
