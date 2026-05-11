# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2843.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 14
- **Target hits / Stop hits / Partials:** 0 / 15 / 2
- **Avg / median % per leg:** -0.94% / -1.83%
- **Sum % (uncompounded):** -15.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.30% | -23.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.30% | -23.0% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 5 | 2 | 1.01% | 7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 0 | 5 | 2 | 1.01% | 7.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 3 | 17.6% | 0 | 15 | 2 | -0.94% | -15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 2740.50 | 2771.79 | 2771.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2723.75 | 2768.07 | 2769.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 2668.00 | 2667.34 | 2705.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 2662.80 | 2667.34 | 2705.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2714.80 | 2652.70 | 2692.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2714.80 | 2652.70 | 2692.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2718.40 | 2653.35 | 2692.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2718.40 | 2653.35 | 2692.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2696.00 | 2655.91 | 2690.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2687.10 | 2655.91 | 2690.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2700.00 | 2656.35 | 2690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 2700.00 | 2656.35 | 2690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2704.50 | 2656.83 | 2690.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 2704.50 | 2656.83 | 2690.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2675.50 | 2656.87 | 2682.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 2681.40 | 2656.87 | 2682.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 2673.20 | 2657.03 | 2682.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 2671.30 | 2656.96 | 2682.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 2537.74 | 2645.27 | 2673.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2684.70 | 2615.80 | 2654.06 | SL hit (close>ema200) qty=0.50 sl=2615.80 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 2831.00 | 2642.74 | 2642.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 14:15:00 | 2836.00 | 2644.67 | 2642.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 2694.20 | 2701.21 | 2676.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:45:00 | 2689.10 | 2701.21 | 2676.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 2682.40 | 2700.98 | 2676.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:45:00 | 2681.60 | 2700.98 | 2676.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 2679.70 | 2700.77 | 2676.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 2675.50 | 2700.77 | 2676.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2680.00 | 2706.54 | 2682.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:45:00 | 2685.20 | 2706.28 | 2682.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2580.00 | 2702.24 | 2680.72 | SL hit (close<static) qty=1.00 sl=2625.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 2437.10 | 2661.85 | 2661.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2431.70 | 2633.85 | 2647.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 2523.90 | 2445.06 | 2523.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 2527.20 | 2445.87 | 2523.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 2527.20 | 2445.87 | 2523.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 2533.20 | 2446.74 | 2523.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:45:00 | 2527.50 | 2446.74 | 2523.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 2512.60 | 2447.40 | 2523.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 2496.50 | 2450.96 | 2523.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 11:15:00 | 2539.70 | 2452.62 | 2523.46 | SL hit (close>static) qty=1.00 sl=2537.90 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 2734.80 | 2572.32 | 2571.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 2749.00 | 2590.63 | 2581.00 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-29 13:15:00 | 2822.50 | 2025-09-30 11:15:00 | 2765.75 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2816.50 | 2025-09-30 11:15:00 | 2765.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-06 09:15:00 | 2824.25 | 2025-10-07 11:15:00 | 2766.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-10-06 13:15:00 | 2819.75 | 2025-10-07 11:15:00 | 2766.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-10-07 13:30:00 | 2792.50 | 2025-10-08 09:15:00 | 2754.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-13 09:15:00 | 2819.50 | 2025-10-24 13:15:00 | 2756.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-24 10:45:00 | 2792.50 | 2025-10-24 13:15:00 | 2756.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-27 09:15:00 | 2795.50 | 2025-10-29 09:15:00 | 2710.50 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-10-28 11:30:00 | 2804.50 | 2025-10-29 09:15:00 | 2710.50 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-05 13:30:00 | 2671.30 | 2026-01-09 12:15:00 | 2537.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 2671.30 | 2026-01-16 09:15:00 | 2684.70 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2655.00 | 2026-01-20 14:15:00 | 2522.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2655.00 | 2026-02-02 14:15:00 | 2575.20 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2026-02-03 14:30:00 | 2668.20 | 2026-02-04 09:15:00 | 2717.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-03 15:00:00 | 2666.60 | 2026-02-04 09:15:00 | 2717.10 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-03-02 10:45:00 | 2685.20 | 2026-03-04 09:15:00 | 2580.00 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-04-09 09:45:00 | 2496.50 | 2026-04-09 11:15:00 | 2539.70 | STOP_HIT | 1.00 | -1.73% |
