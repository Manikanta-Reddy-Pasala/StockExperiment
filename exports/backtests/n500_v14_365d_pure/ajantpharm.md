# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 3033.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 0
- **Avg / median % per leg:** -1.22% / -1.48%
- **Sum % (uncompounded):** -18.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | -0.92% | -5.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | -0.92% | -5.5% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.42% | -12.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.42% | -12.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.22% | -18.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 2746.10 | 2607.04 | 2606.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 12:15:00 | 2775.80 | 2610.16 | 2608.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 2695.50 | 2701.95 | 2664.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:00:00 | 2695.50 | 2701.95 | 2664.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2654.10 | 2701.34 | 2665.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 2696.40 | 2671.38 | 2655.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 2696.70 | 2671.78 | 2656.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2701.70 | 2671.80 | 2656.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 2698.50 | 2672.04 | 2657.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 2655.00 | 2673.61 | 2658.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 2655.00 | 2673.61 | 2658.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 2637.30 | 2673.25 | 2658.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 2609.70 | 2673.25 | 2658.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 2702.30 | 2673.53 | 2658.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 2737.00 | 2673.53 | 2658.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 2627.40 | 2673.27 | 2660.05 | SL hit (close<static) qty=1.00 sl=2630.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 2627.40 | 2673.27 | 2660.05 | SL hit (close<static) qty=1.00 sl=2630.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 2627.40 | 2673.27 | 2660.05 | SL hit (close<static) qty=1.00 sl=2630.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 2627.40 | 2673.27 | 2660.05 | SL hit (close<static) qty=1.00 sl=2630.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 2600.40 | 2667.41 | 2657.86 | SL hit (close<static) qty=1.00 sl=2601.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 2486.40 | 2647.65 | 2648.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 2478.90 | 2645.97 | 2647.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2615.90 | 2609.98 | 2627.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 10:00:00 | 2615.90 | 2609.98 | 2627.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2457.08 | 2501.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 2517.50 | 2457.08 | 2501.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2542.90 | 2457.93 | 2501.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:45:00 | 2541.00 | 2457.93 | 2501.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2515.70 | 2478.74 | 2508.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 2515.70 | 2478.74 | 2508.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2523.10 | 2479.18 | 2508.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 2522.60 | 2479.18 | 2508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2562.20 | 2482.41 | 2508.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 2562.20 | 2482.41 | 2508.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2520.00 | 2497.12 | 2514.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 2500.80 | 2497.37 | 2514.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 2500.00 | 2498.14 | 2514.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:45:00 | 2500.50 | 2498.16 | 2513.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 2509.90 | 2496.15 | 2511.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 2518.40 | 2496.37 | 2511.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 2518.40 | 2496.37 | 2511.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2511.00 | 2496.52 | 2511.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2509.60 | 2496.52 | 2511.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2516.30 | 2496.71 | 2511.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 2494.00 | 2497.67 | 2511.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 2501.20 | 2496.42 | 2510.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 2500.90 | 2496.54 | 2510.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2500.00 | 2496.78 | 2510.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2500.00 | 2496.81 | 2510.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2491.60 | 2496.81 | 2510.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 2521.90 | 2498.03 | 2510.33 | SL hit (close>static) qty=1.00 sl=2521.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2535.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2535.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2535.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2535.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2530.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2530.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2530.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 2537.10 | 2498.42 | 2510.46 | SL hit (close>static) qty=1.00 sl=2530.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 2608.00 | 2520.48 | 2520.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 2636.20 | 2526.96 | 2523.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 2570.80 | 2575.15 | 2552.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 09:45:00 | 2565.30 | 2575.15 | 2552.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2548.10 | 2574.74 | 2552.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2548.10 | 2574.74 | 2552.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2579.50 | 2574.79 | 2552.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:45:00 | 2601.80 | 2575.05 | 2553.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-01 11:15:00 | 2861.98 | 2637.27 | 2593.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 2793.20 | 2840.03 | 2840.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 2758.60 | 2836.32 | 2838.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:00:00 | 2833.30 | 2826.83 | 2832.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 2820.20 | 2826.77 | 2832.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 2839.60 | 2826.77 | 2832.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2918.30 | 2827.68 | 2833.21 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 3062.70 | 2840.16 | 2839.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 3088.70 | 2845.04 | 2841.70 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-12 15:00:00 | 2696.40 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-08-13 13:15:00 | 2696.70 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2701.70 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-18 10:15:00 | 2698.50 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-08-19 15:15:00 | 2737.00 | 2025-08-26 09:15:00 | 2600.40 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-11-14 12:15:00 | 2500.80 | 2025-11-26 10:15:00 | 2521.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-17 09:15:00 | 2500.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-17 14:45:00 | 2500.50 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-19 14:00:00 | 2509.90 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-21 09:15:00 | 2494.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-24 11:15:00 | 2501.20 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-24 14:00:00 | 2500.90 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-24 15:15:00 | 2500.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2491.60 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-12-18 14:45:00 | 2601.80 | 2026-01-01 11:15:00 | 2861.98 | TARGET_HIT | 1.00 | 10.00% |
