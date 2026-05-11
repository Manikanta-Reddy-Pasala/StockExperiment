# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 328.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -2.20% / -2.18%
- **Sum % (uncompounded):** -24.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.69% | -15.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.69% | -15.2% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.52% | -9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.52% | -9.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.20% | -24.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 241.81 | 237.91 | 237.89 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 231.56 | 237.83 | 237.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 229.15 | 237.57 | 237.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 229.61 | 229.29 | 232.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 229.61 | 229.29 | 232.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 236.29 | 229.36 | 232.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 236.29 | 229.36 | 232.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 236.77 | 229.43 | 232.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 236.36 | 229.43 | 232.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 231.43 | 230.69 | 232.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 230.71 | 230.70 | 232.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 230.02 | 230.70 | 232.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 240.77 | 230.83 | 232.49 | SL hit (close>static) qty=1.00 sl=234.29 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 238.04 | 233.94 | 233.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 244.10 | 234.37 | 234.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 233.61 | 235.14 | 234.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 233.61 | 235.14 | 234.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 233.61 | 235.14 | 234.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 234.70 | 235.14 | 234.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 234.70 | 235.14 | 234.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 238.79 | 235.14 | 234.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 231.80 | 235.86 | 235.01 | SL hit (close<static) qty=1.00 sl=232.78 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 225.30 | 234.76 | 234.79 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 248.63 | 234.77 | 234.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 257.25 | 235.13 | 234.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 265.35 | 265.54 | 256.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:00:00 | 265.35 | 265.54 | 256.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 254.65 | 264.60 | 257.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 254.65 | 264.60 | 257.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 251.80 | 264.47 | 257.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 251.80 | 264.47 | 257.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 257.25 | 263.82 | 257.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 257.25 | 263.82 | 257.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 256.40 | 263.75 | 257.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 255.85 | 263.75 | 257.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 256.85 | 263.68 | 257.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:30:00 | 260.00 | 263.66 | 257.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 254.15 | 263.03 | 258.07 | SL hit (close<static) qty=1.00 sl=256.05 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 246.35 | 255.91 | 255.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 245.65 | 255.48 | 255.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 246.60 | 246.24 | 250.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 247.83 | 246.24 | 250.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 250.30 | 244.28 | 248.31 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 273.65 | 250.93 | 250.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 275.70 | 260.52 | 258.12 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-21 11:45:00 | 230.71 | 2025-07-22 11:15:00 | 240.77 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-07-21 14:45:00 | 230.02 | 2025-07-22 11:15:00 | 240.77 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2025-08-04 09:15:00 | 238.79 | 2025-08-07 09:15:00 | 231.80 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-08-12 11:45:00 | 237.16 | 2025-08-22 12:15:00 | 234.88 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-18 12:15:00 | 236.85 | 2025-08-22 12:15:00 | 234.88 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-18 13:15:00 | 236.01 | 2025-08-22 12:15:00 | 234.88 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-19 14:00:00 | 237.90 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-08-20 14:30:00 | 237.13 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-21 09:15:00 | 239.84 | 2025-08-25 10:15:00 | 232.71 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-10-31 13:30:00 | 260.00 | 2025-11-07 09:15:00 | 254.15 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-11-07 14:00:00 | 257.40 | 2025-11-14 12:15:00 | 255.55 | STOP_HIT | 1.00 | -0.72% |
