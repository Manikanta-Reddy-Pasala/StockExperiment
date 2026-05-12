# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 2
- **Avg / median % per leg:** 0.34% / -0.98%
- **Sum % (uncompounded):** 5.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.92% | -19.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.92% | -19.2% |
| SELL (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.85% | 24.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.85% | 24.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 4 | 26.7% | 2 | 11 | 2 | 0.34% | 5.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 256.15 | 235.33 | 235.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 264.20 | 237.96 | 236.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 287.36 | 290.12 | 271.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 285.92 | 290.12 | 271.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 278.95 | 290.49 | 279.06 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 258.43 | 274.41 | 274.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 256.58 | 274.23 | 274.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 272.50 | 267.17 | 270.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 269.35 | 267.19 | 270.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 268.47 | 267.19 | 270.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:45:00 | 261.39 | 267.38 | 270.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:15:00 | 255.05 | 267.29 | 269.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 248.32 | 265.79 | 269.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-26 09:15:00 | 241.62 | 264.33 | 268.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 261.70 | 243.60 | 243.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 263.34 | 244.33 | 243.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 265.38 | 265.38 | 257.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:45:00 | 266.47 | 265.38 | 257.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 258.00 | 265.05 | 258.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 258.00 | 265.05 | 258.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 255.00 | 264.95 | 258.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 254.86 | 264.95 | 258.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 256.75 | 262.29 | 257.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 257.85 | 262.25 | 257.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 258.75 | 262.20 | 257.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 257.86 | 262.11 | 257.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 257.76 | 262.06 | 257.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 257.52 | 262.01 | 257.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:45:00 | 258.60 | 261.99 | 257.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:00:00 | 257.78 | 261.90 | 257.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 258.19 | 261.82 | 257.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:00:00 | 257.80 | 261.78 | 257.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 257.17 | 261.73 | 257.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 257.17 | 261.73 | 257.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 258.42 | 261.70 | 257.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 256.40 | 261.70 | 257.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 255.33 | 261.60 | 257.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 255.33 | 261.60 | 257.51 | SL hit (close<static) qty=1.00 sl=256.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 234.29 | 254.90 | 254.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 233.14 | 253.72 | 254.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 251.62 | 247.20 | 250.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 250.06 | 247.23 | 250.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 250.07 | 247.23 | 250.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 248.86 | 247.25 | 250.38 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 258.27 | 252.91 | 252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 259.76 | 253.34 | 253.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 244.00 | 253.41 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 244.24 | 253.32 | 253.11 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 235.06 | 252.86 | 252.88 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 264.35 | 252.87 | 252.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 268.88 | 254.06 | 253.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 265.07 | 259.37 | 256.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 242.20 | 256.55 | 255.63 | SL hit (close<static) qty=1.00 sl=244.10 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 235.80 | 254.76 | 254.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 232.60 | 254.35 | 254.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 236.80 | 234.89 | 242.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 237.50 | 234.89 | 242.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 242.78 | 233.52 | 240.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:00:00 | 242.78 | 233.52 | 240.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 242.63 | 233.61 | 240.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:15:00 | 243.11 | 233.61 | 240.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 240.40 | 233.72 | 240.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 230.98 | 234.19 | 240.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 244.24 | 234.70 | 240.29 | SL hit (close>static) qty=1.00 sl=243.63 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 280.89 | 244.78 | 244.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 292.59 | 246.31 | 245.53 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-21 12:15:00 | 255.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-25 09:15:00 | 248.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-26 09:15:00 | 241.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-26 11:15:00 | 235.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-11 15:00:00 | 257.85 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 09:15:00 | 258.75 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-12 11:15:00 | 257.86 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 11:45:00 | 257.76 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-12 13:45:00 | 258.60 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-15 10:00:00 | 257.78 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-15 11:45:00 | 258.19 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-15 13:00:00 | 257.80 | 2025-12-16 09:15:00 | 255.33 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-22 14:45:00 | 258.80 | 2025-12-24 13:15:00 | 253.48 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-02-18 09:15:00 | 265.07 | 2026-02-25 12:15:00 | 242.20 | STOP_HIT | 1.00 | -8.63% |
| SELL | retest2 | 2026-04-02 09:15:00 | 230.98 | 2026-04-06 12:15:00 | 244.24 | STOP_HIT | 1.00 | -5.74% |
