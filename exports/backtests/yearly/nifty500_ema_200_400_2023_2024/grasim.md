# Grasim Industries Ltd. (GRASIM)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 48
- **Target hits / Stop hits / Partials:** 4 / 48 / 2
- **Avg / median % per leg:** -0.16% / -1.06%
- **Sum % (uncompounded):** -8.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 2 | 6.1% | 2 | 31 | 0 | -0.41% | -13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 2 | 6.1% | 2 | 31 | 0 | -0.41% | -13.6% |
| SELL (all) | 21 | 4 | 19.0% | 2 | 17 | 2 | 0.24% | 5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 2 | 17 | 2 | 0.24% | 5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 6 | 11.1% | 4 | 48 | 2 | -0.16% | -8.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2583.90 | 2685.27 | 2685.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2575.00 | 2684.18 | 2684.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 2622.10 | 2611.63 | 2641.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:45:00 | 2621.55 | 2611.63 | 2641.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2607.45 | 2612.39 | 2640.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 2598.00 | 2612.14 | 2639.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 2594.85 | 2612.53 | 2639.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 2594.45 | 2612.53 | 2639.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 2598.80 | 2610.69 | 2637.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2657.35 | 2611.06 | 2636.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 2657.35 | 2611.06 | 2636.85 | SL hit (close>static) qty=1.00 sl=2649.45 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 2605.50 | 2471.26 | 2471.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 2626.95 | 2472.80 | 2471.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 12:00:00 | 2485.00 | 2509.12 | 2491.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 2477.85 | 2508.81 | 2491.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:00:00 | 2477.85 | 2508.81 | 2491.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2609.00 | 2691.55 | 2639.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2609.00 | 2691.55 | 2639.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2609.00 | 2690.73 | 2639.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 2610.10 | 2690.73 | 2639.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 2719.60 | 2760.98 | 2718.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 2717.90 | 2760.98 | 2718.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2722.00 | 2760.59 | 2718.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:45:00 | 2736.30 | 2760.35 | 2718.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 2741.00 | 2759.40 | 2718.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 2735.10 | 2758.31 | 2719.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:00:00 | 2742.70 | 2758.16 | 2719.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2727.60 | 2757.38 | 2719.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 2726.20 | 2757.38 | 2719.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 2722.30 | 2756.49 | 2719.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:30:00 | 2727.10 | 2756.49 | 2719.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 2724.00 | 2756.17 | 2719.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 2716.90 | 2756.17 | 2719.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2714.00 | 2755.75 | 2719.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 2714.00 | 2755.75 | 2719.64 | SL hit (close<static) qty=1.00 sl=2716.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.65 | 2803.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 2723.00 | 2800.52 | 2802.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.38 | 2778.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 2763.90 | 2762.38 | 2778.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2769.60 | 2762.46 | 2778.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 2766.80 | 2762.46 | 2778.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2772.60 | 2762.24 | 2777.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:15:00 | 2757.20 | 2762.31 | 2777.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2797.90 | 2762.50 | 2777.20 | SL hit (close>static) qty=1.00 sl=2779.80 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2787.98 | 2787.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.03 | 2790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.82 | 2804.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2801.00 | 2816.66 | 2804.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 2801.00 | 2816.66 | 2804.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2799.30 | 2816.49 | 2804.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2799.30 | 2816.49 | 2804.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2789.70 | 2815.88 | 2804.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 2789.70 | 2815.88 | 2804.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2799.40 | 2815.18 | 2804.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:30:00 | 2810.00 | 2810.99 | 2802.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 2785.90 | 2810.38 | 2802.42 | SL hit (close<static) qty=1.00 sl=2789.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.54 | 2796.63 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.82 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2857.90 | 2798.38 | 2797.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 2795.60 | 2805.26 | 2801.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2762.20 | 2804.83 | 2801.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2769.90 | 2804.83 | 2801.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2833.90 | 2857.65 | 2834.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 2833.90 | 2857.65 | 2834.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2835.00 | 2857.42 | 2834.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 2863.00 | 2857.42 | 2834.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 2810.90 | 2860.22 | 2839.08 | SL hit (close<static) qty=1.00 sl=2822.70 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-09 09:15:00)

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

### Cycle 8 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 2875.50 | 2748.38 | 2747.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2885.00 | 2750.88 | 2749.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 10:45:00 | 1691.23 | 2023-05-26 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-05-25 09:30:00 | 1696.75 | 2023-05-26 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-05-25 13:30:00 | 1690.23 | 2023-05-26 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-05-29 12:30:00 | 1691.03 | 2023-08-08 09:15:00 | 1860.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-27 14:15:00 | 1715.18 | 2023-09-12 10:15:00 | 1886.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-27 12:00:00 | 2598.00 | 2024-12-02 09:15:00 | 2657.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-11-28 10:30:00 | 2594.85 | 2024-12-02 09:15:00 | 2657.35 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-11-28 11:15:00 | 2594.45 | 2024-12-02 09:15:00 | 2657.35 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-11-29 11:15:00 | 2598.80 | 2024-12-02 09:15:00 | 2657.35 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-12-13 10:15:00 | 2632.00 | 2024-12-13 11:15:00 | 2669.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-12-17 10:15:00 | 2638.65 | 2024-12-20 14:15:00 | 2506.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 2630.70 | 2024-12-20 14:15:00 | 2499.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 2638.65 | 2025-01-10 09:15:00 | 2374.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 2630.70 | 2025-01-10 10:15:00 | 2367.63 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-18 13:45:00 | 2736.30 | 2025-07-23 09:15:00 | 2714.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-21 10:15:00 | 2741.00 | 2025-07-23 09:15:00 | 2714.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-21 13:30:00 | 2735.10 | 2025-07-23 09:15:00 | 2714.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-07-21 15:00:00 | 2742.70 | 2025-07-23 09:15:00 | 2714.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-28 10:00:00 | 2748.50 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-28 11:45:00 | 2749.00 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-28 12:15:00 | 2742.40 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-29 14:45:00 | 2745.40 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-04 09:15:00 | 2759.70 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-08-07 11:15:00 | 2732.50 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-07 11:45:00 | 2732.60 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-07 14:15:00 | 2736.10 | 2025-08-08 09:15:00 | 2703.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-11 09:15:00 | 2754.90 | 2025-08-12 09:15:00 | 2722.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-11 09:45:00 | 2753.70 | 2025-08-12 09:15:00 | 2722.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-11 10:45:00 | 2753.90 | 2025-08-12 09:15:00 | 2722.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-11 14:15:00 | 2753.80 | 2025-08-12 09:15:00 | 2722.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-12 13:00:00 | 2730.00 | 2025-11-06 11:15:00 | 2710.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-12 13:30:00 | 2730.00 | 2025-11-06 11:15:00 | 2710.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-13 12:00:00 | 2735.40 | 2025-11-06 11:15:00 | 2710.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-06 10:30:00 | 2729.60 | 2025-11-06 11:15:00 | 2710.20 | STOP_HIT | 1.00 | -0.71% |
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
