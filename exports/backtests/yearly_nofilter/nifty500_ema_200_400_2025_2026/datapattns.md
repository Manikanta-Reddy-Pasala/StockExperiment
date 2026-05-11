# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4118.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 4 |
| TARGET_HIT | 8 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 12
- **Target hits / Stop hits / Partials:** 8 / 13 / 4
- **Avg / median % per leg:** 1.49% / 0.10%
- **Sum % (uncompounded):** 37.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 4 | 4 | 0 | 2.94% | 23.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 2.94% | 23.6% |
| SELL (all) | 17 | 9 | 52.9% | 4 | 9 | 4 | 0.80% | 13.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 9 | 52.9% | 4 | 9 | 4 | 0.80% | 13.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 13 | 52.0% | 8 | 13 | 4 | 1.49% | 37.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 2512.00 | 2685.04 | 2685.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 2505.70 | 2683.25 | 2684.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2574.70 | 2568.39 | 2610.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 2574.70 | 2568.39 | 2610.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2614.20 | 2568.85 | 2609.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:45:00 | 2566.10 | 2569.63 | 2608.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 2570.00 | 2570.12 | 2608.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 2712.60 | 2572.07 | 2608.87 | SL hit (close>static) qty=1.00 sl=2689.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2836.00 | 2638.84 | 2638.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 2843.90 | 2640.88 | 2639.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 2661.40 | 2678.84 | 2660.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2657.00 | 2678.62 | 2660.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 2657.00 | 2678.62 | 2660.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 2642.40 | 2678.26 | 2660.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:30:00 | 2641.20 | 2678.26 | 2660.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2660.00 | 2677.62 | 2660.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:15:00 | 2640.00 | 2677.62 | 2660.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2613.10 | 2676.98 | 2660.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 2616.10 | 2676.98 | 2660.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2709.70 | 2713.73 | 2686.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 2727.40 | 2713.71 | 2687.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 2739.00 | 2743.86 | 2711.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:00:00 | 2741.70 | 2743.31 | 2712.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2636.10 | 2740.07 | 2713.36 | SL hit (close<static) qty=1.00 sl=2685.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2570.30 | 2779.18 | 2779.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2532.00 | 2770.81 | 2775.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 2720.60 | 2711.70 | 2741.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 2730.10 | 2711.70 | 2741.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2716.10 | 2710.70 | 2740.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 12:30:00 | 2708.10 | 2710.70 | 2740.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:15:00 | 2572.69 | 2702.46 | 2733.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2705.50 | 2682.94 | 2719.71 | SL hit (close>ema200) qty=0.50 sl=2682.94 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 2821.90 | 2655.62 | 2655.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 2863.10 | 2662.65 | 2658.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 13:15:00 | 3075.50 | 3076.66 | 2920.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 3075.50 | 3076.66 | 2920.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2958.30 | 3127.96 | 3000.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 2966.80 | 3127.96 | 3000.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2909.00 | 3125.78 | 2999.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:30:00 | 2912.10 | 3125.78 | 2999.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 3069.10 | 3120.95 | 2999.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:45:00 | 3008.00 | 3120.95 | 2999.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 3089.90 | 3119.97 | 3000.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 3119.60 | 3119.58 | 3001.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 3123.60 | 3119.81 | 3001.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 3122.80 | 3120.05 | 3003.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 09:15:00 | 3431.56 | 3155.53 | 3034.79 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-22 09:15:00 | 2762.20 | 2025-07-28 11:15:00 | 2610.70 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2025-09-11 13:45:00 | 2566.10 | 2025-09-12 11:15:00 | 2712.60 | STOP_HIT | 1.00 | -5.71% |
| SELL | retest2 | 2025-09-12 09:30:00 | 2570.00 | 2025-09-12 11:15:00 | 2712.60 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest2 | 2025-10-16 13:15:00 | 2727.40 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-10-31 09:15:00 | 2739.00 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-10-31 13:00:00 | 2741.70 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-11-11 09:15:00 | 2746.80 | 2025-11-13 09:15:00 | 3021.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-26 12:30:00 | 2708.10 | 2025-12-30 12:15:00 | 2572.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 12:30:00 | 2708.10 | 2026-01-05 09:15:00 | 2705.50 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2707.90 | 2026-01-12 11:15:00 | 2572.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:00:00 | 2711.00 | 2026-01-12 11:15:00 | 2575.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 2683.00 | 2026-01-16 09:15:00 | 2548.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2707.90 | 2026-01-20 09:15:00 | 2437.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 14:00:00 | 2711.00 | 2026-01-20 09:15:00 | 2439.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 2683.00 | 2026-01-20 09:15:00 | 2414.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2573.00 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-01-29 11:45:00 | 2576.60 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2026-01-29 14:00:00 | 2571.10 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-02-01 11:45:00 | 2580.00 | 2026-02-01 12:15:00 | 2322.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 2536.20 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.81% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2535.60 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.84% |
| SELL | retest2 | 2026-02-05 09:15:00 | 2537.80 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.75% |
| BUY | retest2 | 2026-04-06 10:30:00 | 3119.60 | 2026-04-10 09:15:00 | 3431.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:30:00 | 3123.60 | 2026-04-10 09:15:00 | 3435.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 3122.80 | 2026-04-10 09:15:00 | 3435.08 | TARGET_HIT | 1.00 | 10.00% |
