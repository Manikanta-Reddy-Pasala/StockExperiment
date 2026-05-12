# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 568.90
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
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -3.73% / -3.09%
- **Sum % (uncompounded):** -37.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.51% | -18.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.51% | -18.1% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.21% | -19.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.21% | -19.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -3.73% | -37.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 245.65 | 224.23 | 224.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 253.69 | 224.72 | 224.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 243.69 | 245.41 | 237.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 243.95 | 245.41 | 237.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 256.20 | 265.47 | 256.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 256.95 | 265.47 | 256.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 256.50 | 265.38 | 256.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 259.10 | 264.22 | 256.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 259.25 | 263.86 | 256.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 261.90 | 263.69 | 256.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 248.60 | 263.52 | 256.41 | SL hit (close<static) qty=1.00 sl=254.55 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 239.80 | 251.54 | 251.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 239.11 | 248.79 | 250.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 243.73 | 242.87 | 246.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 247.00 | 242.96 | 246.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 247.00 | 242.96 | 246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 248.98 | 243.02 | 246.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 249.45 | 243.02 | 246.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 247.51 | 243.21 | 246.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 247.51 | 243.21 | 246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 245.38 | 243.26 | 246.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 245.82 | 243.26 | 246.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 247.54 | 243.26 | 246.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 248.26 | 243.26 | 246.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 246.33 | 243.30 | 246.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 245.01 | 243.31 | 246.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 248.20 | 243.43 | 246.20 | SL hit (close>static) qty=1.00 sl=247.68 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 283.10 | 248.59 | 248.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 289.76 | 249.37 | 248.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 325.45 | 329.01 | 307.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 325.45 | 329.01 | 307.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 314.25 | 331.14 | 316.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 314.25 | 331.14 | 316.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 313.60 | 330.96 | 316.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 318.55 | 330.96 | 316.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 313.80 | 330.64 | 316.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 313.80 | 330.64 | 316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 521.75 | 564.47 | 535.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 521.75 | 564.47 | 535.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 527.00 | 564.10 | 535.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 528.75 | 562.59 | 535.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 503.30 | 561.66 | 534.96 | SL hit (close<static) qty=1.00 sl=519.40 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 458.75 | 516.63 | 516.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 455.00 | 516.01 | 516.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 520.70 | 512.81 | 514.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 520.75 | 512.90 | 514.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 520.80 | 513.00 | 514.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 533.80 | 513.31 | 514.68 | SL hit (close>static) qty=1.00 sl=533.60 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 554.40 | 516.04 | 515.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 561.40 | 518.62 | 517.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 534.40 | 537.37 | 528.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 534.40 | 537.37 | 528.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 535.80 | 537.42 | 529.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 532.85 | 537.42 | 529.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 13:00:00 | 259.10 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-07-30 11:30:00 | 259.25 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-07-30 15:00:00 | 261.90 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-09-08 12:00:00 | 245.01 | 2025-09-10 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-10 15:15:00 | 244.00 | 2025-09-11 10:15:00 | 251.55 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-12 14:45:00 | 528.75 | 2026-03-13 09:15:00 | 503.30 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2026-04-09 13:15:00 | 520.70 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-09 14:15:00 | 520.75 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-09 14:45:00 | 520.80 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-13 09:15:00 | 519.85 | 2026-04-15 09:15:00 | 557.95 | STOP_HIT | 1.00 | -7.33% |
