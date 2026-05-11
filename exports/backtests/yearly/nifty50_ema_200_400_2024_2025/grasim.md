# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2025-11-11 15:15:00 (4484 bars)
- **Last close:** 2771.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 25
- **Target hits / Stop hits / Partials:** 2 / 25 / 2
- **Avg / median % per leg:** -0.13% / -1.15%
- **Sum % (uncompounded):** -3.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.15% | -23.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 0 | 0.0% | 0 | 20 | 0 | -1.15% | -23.0% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 2.13% | 19.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 2 | 5 | 2 | 2.13% | 19.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 4 | 13.8% | 2 | 25 | 2 | -0.13% | -3.8% |

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


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
