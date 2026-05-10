# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 273.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 5 / 3 / 0
- **Avg / median % per leg:** 5.28% / 10.00%
- **Sum % (uncompounded):** 42.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.57% | -7.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.57% | -7.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 5 | 62.5% | 5 | 3 | 0 | 5.28% | 42.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 234.65 | 253.34 | 253.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 234.00 | 253.15 | 253.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 247.64 | 247.56 | 250.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:15:00 | 248.14 | 247.56 | 250.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 251.55 | 247.51 | 250.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 251.55 | 247.51 | 250.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 251.63 | 247.55 | 250.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 251.63 | 247.55 | 250.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 258.50 | 247.69 | 250.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 258.50 | 247.69 | 250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 276.25 | 252.16 | 252.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 280.28 | 252.68 | 252.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 261.67 | 262.06 | 258.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:30:00 | 261.17 | 262.06 | 258.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 259.31 | 261.98 | 258.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 259.31 | 261.98 | 258.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 261.75 | 262.79 | 259.00 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 235.25 | 256.54 | 256.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 233.75 | 251.85 | 254.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 252.70 | 247.16 | 250.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 252.95 | 247.21 | 250.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 251.74 | 247.26 | 250.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 251.82 | 247.36 | 250.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 245.73 | 247.46 | 250.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 256.16 | 247.95 | 250.97 | SL hit (close>static) qty=1.00 sl=255.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 256.16 | 247.95 | 250.97 | SL hit (close>static) qty=1.00 sl=255.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 256.16 | 247.95 | 250.97 | SL hit (close>static) qty=1.00 sl=255.09 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 264.88 | 253.28 | 253.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 267.12 | 255.88 | 254.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:15:00 | 193.69 | 2025-07-09 14:15:00 | 213.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 12:45:00 | 193.32 | 2025-07-09 14:15:00 | 212.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 10:15:00 | 193.55 | 2025-07-09 14:15:00 | 212.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-17 12:15:00 | 193.83 | 2025-07-09 14:15:00 | 213.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-13 09:15:00 | 212.48 | 2025-08-26 14:15:00 | 233.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-06 12:00:00 | 251.74 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-06 13:30:00 | 251.82 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-07 09:15:00 | 245.73 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -4.24% |
