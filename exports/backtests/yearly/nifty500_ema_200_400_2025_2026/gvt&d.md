# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3591 bars)
- **Last close:** 4630.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 7 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 7 / 9 / 0
- **Avg / median % per leg:** 2.77% / -1.13%
- **Sum % (uncompounded):** 44.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 7 | 9 | 0 | 2.77% | 44.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 7 | 43.8% | 7 | 9 | 0 | 2.77% | 44.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 7 | 43.8% | 7 | 9 | 0 | 2.77% | 44.3% |

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

### Cycle 2 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 2634.00 | 2950.62 | 2950.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 2594.80 | 2932.34 | 2941.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 2867.10 | 2852.42 | 2895.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 12:00:00 | 2867.10 | 2852.42 | 2895.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 2880.30 | 2852.69 | 2895.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 2880.30 | 2852.69 | 2895.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 2904.80 | 2853.21 | 2895.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:00:00 | 2904.80 | 2853.21 | 2895.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 2894.20 | 2853.62 | 2895.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:30:00 | 2897.50 | 2853.62 | 2895.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2910.00 | 2854.18 | 2895.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 3046.30 | 2854.18 | 2895.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3004.20 | 2855.67 | 2896.31 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3277.60 | 2933.20 | 2932.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3417.00 | 2941.45 | 2936.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 3606.30 | 3618.98 | 3417.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 3606.30 | 3618.98 | 3417.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 3430.20 | 3615.63 | 3421.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:30:00 | 3400.80 | 3615.63 | 3421.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 3519.00 | 3640.02 | 3464.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 3519.00 | 3640.02 | 3464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 3441.40 | 3638.04 | 3464.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 3434.30 | 3638.04 | 3464.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 3462.60 | 3636.30 | 3464.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 13:45:00 | 3503.30 | 3632.80 | 3464.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 3612.60 | 3629.59 | 3464.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:45:00 | 3490.50 | 3625.78 | 3464.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 12:15:00 | 3490.00 | 3625.78 | 3464.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 09:15:00 | 3853.63 | 3640.30 | 3491.96 | Target hit (10%) qty=1.00 alert=retest2 |


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
| BUY | retest2 | 2026-03-23 13:45:00 | 3503.30 | 2026-04-01 09:15:00 | 3853.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 09:15:00 | 3612.60 | 2026-04-01 09:15:00 | 3839.55 | TARGET_HIT | 1.00 | 6.28% |
| BUY | retest2 | 2026-03-24 11:45:00 | 3490.50 | 2026-04-01 09:15:00 | 3839.00 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2026-03-24 12:15:00 | 3490.00 | 2026-04-01 10:15:00 | 3973.86 | TARGET_HIT | 1.00 | 13.86% |
