# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 328.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 13
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -2.17% / -1.81%
- **Sum % (uncompounded):** -28.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.75% | -14.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.75% | -14.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.85% | -14.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.85% | -14.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.17% | -28.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 240.01 | 232.72 | 232.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 248.05 | 233.05 | 232.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 235.76 | 236.81 | 235.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 12:00:00 | 235.76 | 236.81 | 235.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 233.55 | 236.78 | 235.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 233.55 | 236.78 | 235.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 233.66 | 236.75 | 235.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:15:00 | 232.99 | 236.75 | 235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 232.78 | 236.68 | 235.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 235.83 | 236.68 | 235.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 09:45:00 | 234.59 | 237.67 | 236.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 12:15:00 | 231.56 | 237.53 | 235.96 | SL hit (close<static) qty=1.00 sl=232.22 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 222.81 | 234.59 | 234.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 222.11 | 234.47 | 234.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 229.61 | 229.23 | 231.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 229.61 | 229.23 | 231.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 236.29 | 229.30 | 231.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 236.29 | 229.30 | 231.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 236.77 | 229.37 | 231.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 236.36 | 229.37 | 231.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 231.43 | 230.64 | 231.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 230.71 | 230.65 | 231.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 230.02 | 230.66 | 231.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 240.77 | 230.79 | 231.75 | SL hit (close>static) qty=1.00 sl=234.29 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 245.14 | 232.64 | 232.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 246.22 | 232.77 | 232.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 233.61 | 235.12 | 234.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 233.61 | 235.12 | 234.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 233.61 | 235.12 | 234.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 234.70 | 235.12 | 234.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 234.70 | 235.11 | 234.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 238.79 | 235.11 | 234.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 231.80 | 235.84 | 234.52 | SL hit (close<static) qty=1.00 sl=232.78 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 232.85 | 234.24 | 234.24 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 236.63 | 234.24 | 234.24 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 232.78 | 234.23 | 234.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 229.67 | 234.15 | 234.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 238.22 | 233.62 | 233.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 238.22 | 233.62 | 233.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 238.22 | 233.62 | 233.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 238.22 | 233.62 | 233.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 236.00 | 233.64 | 233.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 234.87 | 233.66 | 233.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 235.27 | 233.70 | 233.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 235.22 | 233.72 | 233.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 239.21 | 233.81 | 234.00 | SL hit (close>static) qty=1.00 sl=238.33 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 243.07 | 234.23 | 234.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 246.38 | 234.35 | 234.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 265.35 | 265.54 | 255.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:00:00 | 265.35 | 265.54 | 255.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 254.65 | 264.59 | 257.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 254.65 | 264.59 | 257.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 251.80 | 264.47 | 257.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 251.80 | 264.47 | 257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 257.25 | 263.82 | 257.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 257.25 | 263.82 | 257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 256.40 | 263.75 | 257.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 255.85 | 263.75 | 257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 256.85 | 263.68 | 257.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:30:00 | 260.00 | 263.66 | 257.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 254.15 | 263.03 | 258.01 | SL hit (close<static) qty=1.00 sl=256.05 alert=retest2 |

### Cycle 8 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 246.35 | 255.91 | 255.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 245.65 | 255.48 | 255.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 246.60 | 246.24 | 250.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 247.83 | 246.24 | 250.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 250.30 | 244.28 | 248.29 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 273.65 | 250.93 | 250.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 275.70 | 260.52 | 258.12 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-04 09:15:00 | 235.83 | 2025-06-13 12:15:00 | 231.56 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-06-13 09:45:00 | 234.59 | 2025-06-13 12:15:00 | 231.56 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-21 11:45:00 | 230.71 | 2025-07-22 11:15:00 | 240.77 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-07-21 14:45:00 | 230.02 | 2025-07-22 11:15:00 | 240.77 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2025-08-04 09:15:00 | 238.79 | 2025-08-07 09:15:00 | 231.80 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-08-12 11:45:00 | 237.16 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-18 12:15:00 | 236.85 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-08-18 13:15:00 | 236.01 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-08 11:45:00 | 234.87 | 2025-09-09 10:15:00 | 239.21 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-08 14:45:00 | 235.27 | 2025-09-09 10:15:00 | 239.21 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-09 09:15:00 | 235.22 | 2025-09-09 10:15:00 | 239.21 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-31 13:30:00 | 260.00 | 2025-11-07 09:15:00 | 254.15 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-11-07 14:00:00 | 257.40 | 2025-11-14 12:15:00 | 255.55 | STOP_HIT | 1.00 | -0.72% |
