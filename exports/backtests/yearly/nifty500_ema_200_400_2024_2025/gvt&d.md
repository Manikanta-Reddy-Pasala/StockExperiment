# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 4630.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 8 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 8 / 15 / 1
- **Avg / median % per leg:** 1.32% / -1.27%
- **Sum % (uncompounded):** 31.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 7 | 9 | 0 | 2.77% | 44.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 7 | 43.8% | 7 | 9 | 0 | 2.77% | 44.3% |
| SELL (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -1.57% | -12.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 1 | 6 | 1 | -1.57% | -12.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 9 | 37.5% | 8 | 15 | 1 | 1.32% | 31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 1590.05 | 1871.39 | 1872.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 1528.30 | 1772.29 | 1814.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1520.30 | 1494.85 | 1602.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 1520.30 | 1494.85 | 1602.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 1596.65 | 1500.81 | 1593.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:00:00 | 1596.65 | 1500.81 | 1593.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1571.45 | 1501.51 | 1593.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:30:00 | 1566.75 | 1502.28 | 1593.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 1622.60 | 1503.48 | 1593.51 | SL hit (close>static) qty=1.00 sl=1598.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 1780.30 | 1554.23 | 1553.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1787.30 | 1558.60 | 1555.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 2277.00 | 2293.19 | 2132.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:45:00 | 2276.10 | 2293.19 | 2132.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2804.20 | 2953.72 | 2835.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2804.20 | 2953.72 | 2835.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2820.00 | 2952.39 | 2835.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 2897.70 | 2952.39 | 2835.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 2838.40 | 2946.34 | 2838.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 11:15:00 | 3122.24 | 2965.81 | 2865.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-19 10:15:00)

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

### Cycle 4 — BUY (started 2026-02-02 14:15:00)

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
| SELL | retest2 | 2025-03-21 13:30:00 | 1566.75 | 2025-03-21 14:15:00 | 1622.60 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-03-25 12:30:00 | 1564.50 | 2025-03-27 14:15:00 | 1616.20 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-03-26 11:00:00 | 1565.10 | 2025-03-27 14:15:00 | 1616.20 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-03-27 13:45:00 | 1554.95 | 2025-03-27 14:15:00 | 1616.20 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1528.15 | 2025-04-03 09:15:00 | 1451.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1528.15 | 2025-04-07 09:15:00 | 1375.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 15:15:00 | 1526.00 | 2025-05-07 14:15:00 | 1630.90 | STOP_HIT | 1.00 | -6.87% |
| SELL | retest2 | 2025-05-07 09:45:00 | 1529.60 | 2025-05-07 14:15:00 | 1630.90 | STOP_HIT | 1.00 | -6.62% |
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
