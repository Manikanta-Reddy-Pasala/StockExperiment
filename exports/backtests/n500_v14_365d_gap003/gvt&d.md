# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4630.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 0
- **Avg / median % per leg:** 0.35% / -1.27%
- **Sum % (uncompounded):** 4.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.35% | 4.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.35% | 4.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 3 | 25.0% | 3 | 9 | 0 | 0.35% | 4.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 1780.30 | 1554.23 | 1553.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1787.30 | 1558.60 | 1555.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 2277.00 | 2293.19 | 2132.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:45:00 | 2276.10 | 2293.19 | 2132.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2804.20 | 2953.72 | 2835.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2804.20 | 2953.72 | 2835.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2820.00 | 2952.39 | 2835.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 2897.70 | 2952.39 | 2835.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 2838.40 | 2946.34 | 2838.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 11:15:00 | 3122.24 | 2965.81 | 2865.47 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-11-03 13:15:00 | 3187.47 | 2970.78 | 2868.96 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 2839.70 | 2985.14 | 2935.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 2803.50 | 2981.76 | 2933.89 | SL hit (close<static) qty=1.00 sl=2804.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 2848.00 | 2979.95 | 2933.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 2900.00 | 2969.85 | 2930.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 2879.40 | 2969.85 | 2930.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 2800.00 | 2958.45 | 2926.40 | SL hit (close<static) qty=1.00 sl=2804.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 2905.50 | 2925.55 | 2912.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 2904.00 | 2925.55 | 2912.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2914.00 | 2925.44 | 2912.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:30:00 | 2897.20 | 2925.44 | 2912.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 2896.00 | 2925.15 | 2912.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2965.80 | 2925.15 | 2912.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 2920.90 | 2925.54 | 2912.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 15:15:00 | 2888.00 | 2925.05 | 2912.44 | SL hit (close<static) qty=1.00 sl=2891.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 15:15:00 | 2888.00 | 2925.05 | 2912.44 | SL hit (close<static) qty=1.00 sl=2891.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:45:00 | 2919.90 | 2924.03 | 2912.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 2846.40 | 2943.98 | 2924.88 | SL hit (close<static) qty=1.00 sl=2891.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 2927.40 | 2938.44 | 2922.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2897.50 | 2938.03 | 2922.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2897.50 | 2938.03 | 2922.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2899.00 | 2937.64 | 2922.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 2915.00 | 2937.64 | 2922.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 2891.00 | 2936.91 | 2922.40 | SL hit (close<static) qty=1.00 sl=2891.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 3190.00 | 2936.91 | 2922.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-22 09:15:00 | 3206.50 | 2938.39 | 2923.21 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 2905.80 | 3038.38 | 2990.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 2865.00 | 3036.66 | 2989.68 | SL hit (close<static) qty=1.00 sl=2876.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 2865.00 | 3036.66 | 2989.68 | SL hit (close<static) qty=1.00 sl=2876.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 2903.30 | 3031.80 | 2987.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 2792.20 | 3026.93 | 2986.14 | SL hit (close<static) qty=1.00 sl=2876.00 alert=retest2 |
| CROSSOVER_SKIP | 2026-01-19 10:15:00 | 2634.00 | 2950.62 | 2950.89 | min_gap filter: gap=0.010% < 0.030% |
| TREND_RESET | 2026-01-19 10:15:00 | 2634.00 | 2950.62 | 2950.89 | EMA inversion without crossover edge (EMA200=2950.62 EMA400=2950.89) — end cycle |
| CROSSOVER_SKIP | 2026-02-02 14:15:00 | 3277.60 | 2933.20 | 2932.47 | min_gap filter: gap=0.022% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-24 09:15:00 | 2897.70 | 2025-11-03 11:15:00 | 3122.24 | TARGET_HIT | 1.00 | 7.75% |
| BUY | retest2 | 2025-10-27 12:15:00 | 2838.40 | 2025-11-03 13:15:00 | 3187.47 | TARGET_HIT | 1.00 | 12.30% |
| BUY | retest2 | 2025-12-01 12:45:00 | 2839.70 | 2025-12-01 14:15:00 | 2803.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-02 09:15:00 | 2848.00 | 2025-12-04 14:15:00 | 2800.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-10 09:15:00 | 2965.80 | 2025-12-10 15:15:00 | 2888.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-12-10 13:30:00 | 2920.90 | 2025-12-10 15:15:00 | 2888.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-11 12:45:00 | 2919.90 | 2025-12-18 09:15:00 | 2846.40 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-12-19 10:45:00 | 2927.40 | 2025-12-19 15:15:00 | 2891.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-19 13:15:00 | 2915.00 | 2025-12-22 09:15:00 | 3206.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-22 09:15:00 | 3190.00 | 2026-01-09 10:15:00 | 2865.00 | STOP_HIT | 1.00 | -10.19% |
| BUY | retest2 | 2026-01-09 09:45:00 | 2905.80 | 2026-01-09 10:15:00 | 2865.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-01-09 13:45:00 | 2903.30 | 2026-01-12 09:15:00 | 2792.20 | STOP_HIT | 1.00 | -3.83% |
