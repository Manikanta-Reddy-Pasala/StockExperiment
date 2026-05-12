# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 339.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 54 |
| ALERT2 | 52 |
| ALERT2_SKIP | 26 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 61 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 45
- **Target hits / Stop hits / Partials:** 0 / 63 / 1
- **Avg / median % per leg:** -0.94% / -0.76%
- **Sum % (uncompounded):** -60.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 10 | 30.3% | 0 | 33 | 0 | -1.31% | -43.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 10 | 30.3% | 0 | 33 | 0 | -1.31% | -43.3% |
| SELL (all) | 31 | 9 | 29.0% | 0 | 30 | 1 | -0.55% | -16.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.05% | -2.1% |
| SELL @ 3rd Alert (retest2) | 29 | 9 | 31.0% | 0 | 28 | 1 | -0.51% | -14.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.05% | -2.1% |
| retest2 (combined) | 62 | 19 | 30.6% | 0 | 61 | 1 | -0.94% | -58.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 258.70 | 254.06 | 253.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 261.70 | 255.59 | 254.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 256.60 | 257.14 | 255.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 256.60 | 257.14 | 255.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 255.20 | 256.75 | 255.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 255.20 | 256.75 | 255.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 255.65 | 256.53 | 255.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 257.45 | 256.61 | 255.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 258.15 | 257.28 | 256.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:30:00 | 258.40 | 257.53 | 256.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 258.80 | 263.34 | 263.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 258.80 | 263.34 | 263.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 258.55 | 261.88 | 263.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 260.00 | 259.77 | 261.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 260.00 | 259.77 | 261.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 257.25 | 256.67 | 257.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 258.40 | 256.67 | 257.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 257.00 | 256.74 | 257.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:30:00 | 256.20 | 257.01 | 257.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 256.25 | 256.94 | 257.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 258.15 | 257.06 | 257.18 | SL hit (close>static) qty=1.00 sl=257.65 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 258.40 | 257.33 | 257.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 262.90 | 258.54 | 257.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 261.40 | 262.33 | 260.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 261.40 | 262.33 | 260.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 261.70 | 262.60 | 261.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 261.75 | 262.60 | 261.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 261.85 | 262.45 | 261.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 261.85 | 262.45 | 261.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 261.60 | 262.28 | 261.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 260.75 | 262.28 | 261.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 262.00 | 262.23 | 261.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 262.70 | 262.23 | 261.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 261.55 | 262.09 | 261.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 261.55 | 262.09 | 261.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 260.35 | 261.74 | 261.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 259.25 | 261.74 | 261.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 259.85 | 261.36 | 261.47 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 262.55 | 261.08 | 261.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 274.10 | 265.18 | 263.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 281.65 | 282.23 | 277.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 281.65 | 282.23 | 277.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 282.65 | 281.77 | 279.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 279.85 | 281.77 | 279.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 281.20 | 281.51 | 280.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 281.80 | 281.49 | 280.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 277.60 | 280.71 | 280.05 | SL hit (close<static) qty=1.00 sl=280.05 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 275.80 | 278.91 | 279.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 273.45 | 277.82 | 278.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 271.45 | 271.28 | 273.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 270.15 | 271.28 | 273.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 10:30:00 | 270.75 | 271.04 | 273.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 273.30 | 271.85 | 273.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 273.30 | 271.85 | 273.17 | SL hit (close>ema400) qty=1.00 sl=273.17 alert=retest1 |

### Cycle 7 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 267.60 | 263.49 | 263.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 269.00 | 264.59 | 263.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 267.50 | 267.85 | 266.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:45:00 | 267.75 | 267.85 | 266.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 266.50 | 267.58 | 266.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 266.50 | 267.58 | 266.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 266.50 | 267.36 | 266.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 266.50 | 267.36 | 266.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 266.40 | 267.17 | 266.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 268.15 | 267.17 | 266.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 267.15 | 267.13 | 266.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 266.80 | 267.06 | 266.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 267.45 | 269.64 | 269.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 267.45 | 269.64 | 269.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 266.20 | 268.96 | 269.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 268.30 | 264.15 | 265.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 268.30 | 264.15 | 265.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 268.30 | 264.15 | 265.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 268.30 | 264.15 | 265.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 270.15 | 265.35 | 266.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 273.00 | 265.35 | 266.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 268.80 | 266.76 | 266.60 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 265.00 | 266.32 | 266.50 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 268.55 | 266.75 | 266.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 269.25 | 268.01 | 267.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 269.80 | 269.88 | 269.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 269.80 | 269.88 | 269.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 268.95 | 269.70 | 269.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 268.95 | 269.70 | 269.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 266.50 | 269.06 | 268.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 266.50 | 269.06 | 268.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 266.45 | 268.54 | 268.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 265.35 | 267.59 | 268.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 10:15:00 | 267.80 | 266.52 | 267.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 10:15:00 | 267.80 | 266.52 | 267.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 267.80 | 266.52 | 267.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 267.80 | 266.52 | 267.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 267.55 | 266.73 | 267.13 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 268.25 | 267.39 | 267.34 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 264.90 | 266.89 | 267.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 12:15:00 | 262.30 | 263.74 | 264.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 260.25 | 260.23 | 261.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 260.25 | 260.23 | 261.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 262.50 | 260.68 | 261.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 262.30 | 260.68 | 261.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 262.15 | 260.98 | 261.62 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 263.10 | 261.98 | 261.96 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 261.15 | 261.81 | 261.89 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 265.00 | 262.28 | 262.05 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 259.40 | 261.64 | 261.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 258.10 | 260.08 | 261.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 258.60 | 257.99 | 259.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 258.60 | 257.99 | 259.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 259.15 | 258.42 | 259.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 258.50 | 258.42 | 259.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 257.75 | 258.52 | 259.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 256.90 | 252.72 | 252.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 256.90 | 252.72 | 252.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 257.20 | 253.62 | 253.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 253.75 | 255.22 | 254.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 253.75 | 255.22 | 254.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 253.75 | 255.22 | 254.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 257.90 | 255.80 | 254.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 261.00 | 256.94 | 255.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 255.70 | 256.73 | 256.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 255.70 | 256.73 | 256.75 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 257.30 | 256.85 | 256.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 258.75 | 257.23 | 256.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 257.00 | 257.34 | 257.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 257.00 | 257.34 | 257.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 257.00 | 257.34 | 257.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 257.00 | 257.34 | 257.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 256.75 | 257.22 | 257.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 256.60 | 257.22 | 257.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 256.85 | 257.15 | 257.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:30:00 | 256.80 | 257.15 | 257.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 257.30 | 257.18 | 257.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:15:00 | 257.10 | 257.18 | 257.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 256.15 | 256.97 | 256.97 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 257.00 | 256.98 | 256.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 255.85 | 256.75 | 256.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 254.65 | 256.33 | 256.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 253.05 | 252.94 | 254.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:30:00 | 252.85 | 252.94 | 254.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 254.85 | 253.32 | 254.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 256.50 | 253.32 | 254.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 255.70 | 253.80 | 254.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:15:00 | 257.65 | 253.80 | 254.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 255.40 | 254.12 | 254.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 254.75 | 254.12 | 254.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 254.85 | 254.43 | 254.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 254.25 | 252.88 | 253.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 256.30 | 253.78 | 253.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 256.30 | 253.78 | 253.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 257.00 | 255.23 | 254.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 257.05 | 258.10 | 256.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 257.05 | 258.10 | 256.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 257.05 | 258.10 | 256.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 256.85 | 258.10 | 256.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 256.65 | 257.81 | 256.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 256.65 | 257.81 | 256.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 256.75 | 257.60 | 256.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 256.60 | 257.60 | 256.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 256.25 | 257.33 | 256.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:15:00 | 256.35 | 257.33 | 256.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 255.35 | 256.93 | 256.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 255.00 | 256.93 | 256.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 255.00 | 256.24 | 256.40 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 265.70 | 258.13 | 257.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 267.60 | 261.13 | 258.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 261.85 | 263.99 | 261.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 261.85 | 263.99 | 261.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 261.85 | 263.99 | 261.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 261.85 | 263.99 | 261.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 261.40 | 263.48 | 261.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 260.70 | 263.48 | 261.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 261.85 | 263.15 | 261.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:30:00 | 261.80 | 263.15 | 261.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 262.00 | 262.92 | 261.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 261.30 | 262.92 | 261.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 263.40 | 262.98 | 261.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 262.30 | 263.03 | 262.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 264.25 | 263.22 | 262.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 263.85 | 263.22 | 262.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 261.40 | 264.42 | 263.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 261.40 | 264.42 | 263.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 260.30 | 263.59 | 263.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 260.30 | 263.59 | 263.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 260.65 | 263.01 | 263.22 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 264.15 | 263.42 | 263.37 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 262.55 | 263.31 | 263.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 262.45 | 263.13 | 263.26 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 264.45 | 263.40 | 263.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 266.50 | 264.02 | 263.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 265.65 | 265.69 | 264.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 09:15:00 | 262.85 | 265.69 | 264.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 263.65 | 265.29 | 264.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 262.35 | 265.29 | 264.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 264.20 | 265.07 | 264.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 263.75 | 265.07 | 264.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 261.60 | 264.02 | 264.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 260.50 | 263.05 | 263.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 261.45 | 260.54 | 261.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 261.45 | 260.54 | 261.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 261.45 | 260.54 | 261.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 261.45 | 260.54 | 261.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 259.00 | 260.23 | 261.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 260.75 | 260.23 | 261.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 255.75 | 259.09 | 260.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:15:00 | 255.45 | 259.09 | 260.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 255.35 | 258.03 | 260.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 255.50 | 255.77 | 257.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 260.15 | 258.29 | 258.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 13:15:00 | 260.15 | 258.29 | 258.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 261.05 | 258.84 | 258.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 259.60 | 260.48 | 259.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 259.60 | 260.48 | 259.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 259.60 | 260.48 | 259.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 259.60 | 260.48 | 259.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 259.80 | 260.34 | 259.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 263.65 | 260.34 | 259.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 267.70 | 269.30 | 269.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 11:15:00 | 267.70 | 269.30 | 269.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 276.50 | 270.80 | 270.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 278.30 | 274.19 | 272.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 283.85 | 284.36 | 280.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:00:00 | 283.85 | 284.36 | 280.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 290.05 | 289.11 | 287.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 289.75 | 289.11 | 287.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 285.20 | 288.18 | 287.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 285.20 | 288.18 | 287.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 284.50 | 287.44 | 286.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 284.50 | 287.44 | 286.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 283.90 | 286.20 | 286.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 283.50 | 285.66 | 286.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 10:15:00 | 285.90 | 284.59 | 285.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 285.90 | 284.59 | 285.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 285.90 | 284.59 | 285.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 285.90 | 284.59 | 285.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 285.25 | 284.72 | 285.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:30:00 | 284.95 | 284.72 | 285.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 284.65 | 284.71 | 285.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 283.30 | 284.26 | 284.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 277.50 | 274.90 | 274.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 277.50 | 274.90 | 274.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 280.20 | 277.02 | 275.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 280.00 | 280.17 | 278.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 280.00 | 280.17 | 278.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 282.60 | 280.68 | 278.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 279.00 | 280.68 | 278.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 277.85 | 280.73 | 279.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 277.85 | 280.73 | 279.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 280.10 | 280.60 | 279.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 277.85 | 280.60 | 279.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 278.00 | 280.08 | 279.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 278.00 | 280.08 | 279.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 277.35 | 279.53 | 279.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 277.35 | 279.53 | 279.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 276.85 | 279.00 | 279.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 274.20 | 277.50 | 278.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 277.25 | 276.00 | 277.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 277.25 | 276.00 | 277.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 277.25 | 276.00 | 277.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 277.25 | 276.00 | 277.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 281.40 | 277.08 | 277.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 281.40 | 277.08 | 277.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 279.80 | 277.62 | 277.86 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 280.35 | 278.17 | 278.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 282.10 | 280.20 | 279.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 280.40 | 280.43 | 279.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 280.40 | 280.43 | 279.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 280.40 | 280.43 | 279.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 280.20 | 280.43 | 279.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 296.55 | 297.73 | 296.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 298.90 | 297.49 | 296.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 298.00 | 298.03 | 296.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:30:00 | 298.00 | 298.03 | 296.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:45:00 | 298.00 | 298.01 | 297.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 297.45 | 298.55 | 297.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 297.45 | 298.55 | 297.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 298.10 | 298.46 | 297.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 297.00 | 298.46 | 297.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 298.05 | 298.38 | 297.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 298.05 | 298.38 | 297.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 298.00 | 298.30 | 297.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 297.55 | 298.30 | 297.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 297.30 | 298.10 | 297.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 298.00 | 298.10 | 297.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 296.60 | 297.56 | 297.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 296.60 | 297.56 | 297.66 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 299.05 | 297.88 | 297.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 299.75 | 298.43 | 298.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 299.20 | 299.25 | 298.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 299.20 | 299.25 | 298.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 299.20 | 299.25 | 298.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 299.25 | 299.25 | 298.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 317.00 | 318.28 | 317.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 317.00 | 318.28 | 317.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 312.80 | 317.18 | 316.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 312.80 | 317.18 | 316.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 314.10 | 316.57 | 316.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 304.95 | 312.06 | 314.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 311.75 | 309.94 | 312.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 311.25 | 309.94 | 312.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 310.85 | 310.12 | 312.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 311.50 | 310.12 | 312.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 311.85 | 310.47 | 312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 313.95 | 310.47 | 312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 317.70 | 311.91 | 312.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:00:00 | 311.25 | 312.66 | 312.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 311.50 | 312.39 | 312.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 318.00 | 313.52 | 313.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 318.00 | 313.52 | 313.19 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 309.65 | 312.71 | 313.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 15:15:00 | 308.95 | 311.07 | 312.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 310.95 | 310.46 | 311.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 310.95 | 310.46 | 311.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 309.65 | 310.03 | 310.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 311.05 | 310.03 | 310.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 311.60 | 310.34 | 310.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 311.60 | 310.34 | 310.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 309.20 | 310.11 | 310.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 308.40 | 309.51 | 310.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 308.05 | 309.51 | 310.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 313.20 | 310.01 | 310.40 | SL hit (close>static) qty=1.00 sl=312.15 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 313.45 | 311.15 | 310.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 314.40 | 312.46 | 311.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 343.35 | 344.30 | 338.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 15:00:00 | 343.35 | 344.30 | 338.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 350.65 | 355.97 | 353.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 350.65 | 355.97 | 353.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 349.60 | 354.69 | 353.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 349.80 | 354.69 | 353.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 360.10 | 369.33 | 365.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 358.80 | 369.33 | 365.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 360.40 | 367.55 | 365.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 360.70 | 367.55 | 365.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 368.50 | 367.71 | 365.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 373.50 | 367.71 | 365.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 370.65 | 368.30 | 366.19 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 359.50 | 364.99 | 365.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 354.50 | 362.90 | 364.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 357.85 | 351.24 | 354.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 357.85 | 351.24 | 354.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 357.85 | 351.24 | 354.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 357.85 | 351.24 | 354.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 366.65 | 354.32 | 355.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 366.65 | 354.32 | 355.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 362.20 | 355.90 | 356.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 366.30 | 355.90 | 356.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 363.70 | 357.46 | 357.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 366.40 | 360.45 | 358.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 357.50 | 361.00 | 359.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 357.50 | 361.00 | 359.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 357.50 | 361.00 | 359.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 357.50 | 361.00 | 359.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 356.60 | 360.12 | 358.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 356.60 | 360.12 | 358.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 351.90 | 358.47 | 358.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 351.90 | 358.47 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 348.30 | 356.44 | 357.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 347.25 | 350.16 | 352.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 348.80 | 348.48 | 350.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 348.80 | 348.48 | 350.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 348.80 | 348.48 | 350.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 349.80 | 348.48 | 350.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 350.00 | 344.44 | 346.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 352.00 | 344.44 | 346.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 349.70 | 345.49 | 347.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:30:00 | 348.40 | 346.20 | 347.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 347.25 | 346.89 | 347.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 14:00:00 | 347.75 | 347.06 | 347.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 348.30 | 347.00 | 347.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 348.30 | 347.26 | 347.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 348.30 | 347.26 | 347.14 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 345.85 | 346.98 | 347.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 342.90 | 345.30 | 346.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 349.75 | 345.30 | 345.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 349.75 | 345.30 | 345.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 349.75 | 345.30 | 345.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 351.25 | 345.30 | 345.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 350.45 | 346.33 | 346.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 352.55 | 349.03 | 347.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 387.30 | 388.18 | 379.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 387.30 | 388.18 | 379.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 381.25 | 386.79 | 379.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 377.60 | 386.79 | 379.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 386.55 | 391.49 | 389.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 385.55 | 391.49 | 389.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 386.40 | 390.47 | 389.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 387.30 | 390.47 | 389.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 383.40 | 388.10 | 388.22 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 392.25 | 388.80 | 388.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 398.80 | 390.80 | 389.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 399.65 | 399.66 | 395.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 399.40 | 399.66 | 395.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 400.75 | 401.98 | 399.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 399.50 | 401.98 | 399.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 401.70 | 403.91 | 402.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 392.90 | 403.91 | 402.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 390.40 | 401.21 | 401.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 388.80 | 398.73 | 400.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 367.00 | 364.78 | 372.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:30:00 | 368.20 | 364.78 | 372.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 354.30 | 351.89 | 354.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 355.50 | 351.89 | 354.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 353.45 | 352.20 | 354.85 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 359.80 | 356.33 | 356.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 363.10 | 357.69 | 356.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 354.55 | 357.82 | 357.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 354.55 | 357.82 | 357.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 354.55 | 357.82 | 357.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 354.55 | 357.82 | 357.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 357.60 | 357.78 | 357.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 359.45 | 357.99 | 357.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 354.10 | 356.54 | 356.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 354.10 | 356.54 | 356.85 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 359.90 | 357.21 | 356.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 11:15:00 | 361.15 | 357.99 | 357.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 359.10 | 359.57 | 358.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 359.10 | 359.57 | 358.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 359.10 | 359.57 | 358.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 359.20 | 359.57 | 358.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 358.45 | 359.34 | 358.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 358.45 | 359.34 | 358.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 357.90 | 359.06 | 358.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 359.35 | 359.06 | 358.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 356.30 | 358.50 | 358.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 356.30 | 358.50 | 358.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 353.75 | 357.55 | 357.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 352.10 | 356.46 | 357.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 351.05 | 350.93 | 353.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 361.80 | 350.93 | 353.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 359.20 | 352.58 | 353.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 361.00 | 352.58 | 353.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 359.20 | 353.91 | 354.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:00:00 | 356.00 | 354.33 | 354.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 360.35 | 355.53 | 355.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 360.35 | 355.53 | 355.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 362.95 | 358.88 | 356.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 361.00 | 361.45 | 359.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:30:00 | 360.80 | 361.45 | 359.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 368.65 | 362.82 | 360.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 370.70 | 364.41 | 362.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:00:00 | 370.35 | 365.59 | 362.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:45:00 | 370.10 | 366.48 | 363.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 13:00:00 | 369.80 | 367.14 | 364.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 361.60 | 367.46 | 365.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 374.55 | 368.16 | 366.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:45:00 | 376.70 | 376.00 | 371.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 374.20 | 376.77 | 372.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:30:00 | 374.50 | 375.81 | 372.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 372.65 | 375.54 | 373.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 372.65 | 375.54 | 373.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 362.40 | 372.91 | 372.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 364.10 | 372.91 | 372.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 358.05 | 369.94 | 371.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 358.05 | 369.94 | 371.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 352.50 | 363.38 | 367.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 361.00 | 354.50 | 360.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 361.00 | 354.50 | 360.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 361.00 | 354.50 | 360.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 371.00 | 354.50 | 360.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 365.35 | 356.67 | 360.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 361.20 | 359.57 | 361.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 361.00 | 359.57 | 361.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:45:00 | 360.70 | 360.18 | 361.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 366.85 | 362.20 | 361.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 366.85 | 362.20 | 361.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 368.70 | 365.05 | 363.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 372.80 | 373.37 | 370.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 372.80 | 373.37 | 370.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 372.80 | 373.37 | 370.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 377.80 | 373.83 | 370.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 376.55 | 376.03 | 372.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 377.65 | 375.85 | 372.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 378.90 | 383.75 | 384.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 378.90 | 383.75 | 384.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 370.75 | 378.43 | 381.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 368.80 | 367.10 | 371.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 368.80 | 367.10 | 371.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 371.25 | 367.93 | 371.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 373.95 | 367.93 | 371.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 383.70 | 371.09 | 372.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 383.70 | 371.09 | 372.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 384.60 | 373.79 | 373.38 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 375.60 | 379.15 | 379.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 373.50 | 377.18 | 378.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 378.80 | 376.90 | 378.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 378.80 | 376.90 | 378.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 378.80 | 376.90 | 378.10 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 381.75 | 378.52 | 378.45 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 365.75 | 375.96 | 377.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 364.10 | 373.59 | 376.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 376.00 | 369.20 | 372.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 376.00 | 369.20 | 372.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 376.00 | 369.20 | 372.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 376.00 | 369.20 | 372.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 376.25 | 370.61 | 372.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 376.55 | 370.61 | 372.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 381.45 | 374.76 | 374.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 386.80 | 380.96 | 378.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 377.70 | 381.11 | 378.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 377.70 | 381.11 | 378.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 377.70 | 381.11 | 378.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 377.70 | 381.11 | 378.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 379.50 | 380.79 | 378.92 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 373.70 | 377.39 | 377.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 364.80 | 374.88 | 376.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 364.70 | 363.88 | 366.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 364.70 | 363.88 | 366.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 365.00 | 364.00 | 365.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 367.40 | 364.00 | 365.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 364.40 | 364.08 | 365.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 346.65 | 364.03 | 364.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 361.70 | 355.46 | 356.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:45:00 | 361.80 | 357.48 | 357.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 15:15:00 | 360.40 | 358.06 | 357.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 360.40 | 358.06 | 357.90 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 355.25 | 357.62 | 357.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 354.65 | 357.02 | 357.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 322.40 | 319.91 | 326.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 14:00:00 | 322.40 | 319.91 | 326.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 322.50 | 320.44 | 324.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 324.20 | 320.44 | 324.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 323.15 | 321.44 | 324.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 324.55 | 321.44 | 324.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 307.40 | 299.14 | 303.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 307.40 | 299.14 | 303.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 304.65 | 300.24 | 303.41 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 315.60 | 305.05 | 304.98 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 301.75 | 308.30 | 308.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 292.60 | 304.07 | 306.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 296.75 | 293.21 | 298.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 296.75 | 293.21 | 298.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 296.75 | 293.21 | 298.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 290.85 | 292.85 | 296.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 12:15:00 | 276.31 | 284.36 | 290.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 285.50 | 284.37 | 289.39 | SL hit (close>ema200) qty=0.50 sl=284.37 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 297.40 | 284.26 | 283.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 299.60 | 296.27 | 292.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 292.40 | 297.98 | 295.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 292.40 | 297.98 | 295.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 292.40 | 297.98 | 295.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 298.00 | 296.69 | 295.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 304.20 | 296.73 | 295.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:00:00 | 298.05 | 299.15 | 298.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 298.15 | 298.43 | 298.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 294.30 | 297.60 | 297.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 294.30 | 297.60 | 297.68 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 299.20 | 297.66 | 297.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 300.30 | 298.19 | 297.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 13:15:00 | 297.95 | 298.33 | 298.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 13:15:00 | 297.95 | 298.33 | 298.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 297.95 | 298.33 | 298.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:00:00 | 297.95 | 298.33 | 298.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 298.90 | 298.45 | 298.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 15:15:00 | 300.00 | 298.45 | 298.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 301.05 | 299.22 | 298.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 300.10 | 302.51 | 302.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 300.10 | 302.51 | 302.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 297.45 | 301.29 | 302.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 294.55 | 294.49 | 296.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 326.30 | 294.49 | 296.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 77 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 314.75 | 298.54 | 298.36 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 311.45 | 314.07 | 314.21 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 318.00 | 314.86 | 314.56 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 311.25 | 313.86 | 314.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 308.60 | 312.80 | 313.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 304.65 | 304.64 | 308.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 304.65 | 304.64 | 308.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 305.90 | 304.85 | 307.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 314.20 | 304.85 | 307.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 317.50 | 307.38 | 308.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 315.75 | 307.38 | 308.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 319.75 | 309.86 | 309.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 321.20 | 312.12 | 310.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 11:00:00 | 257.45 | 2025-05-20 11:15:00 | 258.80 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-05-14 13:00:00 | 258.15 | 2025-05-20 11:15:00 | 258.80 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-05-14 13:30:00 | 258.40 | 2025-05-20 11:15:00 | 258.80 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-05-26 14:30:00 | 256.20 | 2025-05-27 13:15:00 | 258.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-27 09:15:00 | 256.25 | 2025-05-27 13:15:00 | 258.15 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-12 09:45:00 | 281.80 | 2025-06-12 10:15:00 | 277.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2025-06-16 09:15:00 | 270.15 | 2025-06-16 13:15:00 | 273.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2025-06-16 10:30:00 | 270.75 | 2025-06-16 13:15:00 | 273.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-06-17 10:45:00 | 272.00 | 2025-06-24 10:15:00 | 267.60 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-06-26 09:15:00 | 268.15 | 2025-07-01 10:15:00 | 267.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-06-26 12:00:00 | 267.15 | 2025-07-01 10:15:00 | 267.45 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-06-26 13:00:00 | 266.80 | 2025-07-01 10:15:00 | 267.45 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-25 09:15:00 | 258.50 | 2025-07-30 10:15:00 | 256.90 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-07-25 10:15:00 | 257.75 | 2025-07-30 10:15:00 | 256.90 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-31 11:45:00 | 257.90 | 2025-08-04 12:15:00 | 255.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-08-01 09:15:00 | 261.00 | 2025-08-04 12:15:00 | 255.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-08 11:15:00 | 254.75 | 2025-08-12 09:15:00 | 256.30 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-08 13:00:00 | 254.85 | 2025-08-12 09:15:00 | 256.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-08-11 15:00:00 | 254.25 | 2025-08-12 09:15:00 | 256.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-29 10:15:00 | 255.45 | 2025-09-02 13:15:00 | 260.15 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-08-29 12:15:00 | 255.35 | 2025-09-02 13:15:00 | 260.15 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-09-01 10:15:00 | 255.50 | 2025-09-02 13:15:00 | 260.15 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-04 09:15:00 | 263.65 | 2025-09-15 11:15:00 | 267.70 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-09-24 14:30:00 | 283.30 | 2025-10-06 09:15:00 | 277.50 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2025-10-21 14:15:00 | 298.90 | 2025-10-27 12:15:00 | 296.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-23 09:45:00 | 298.00 | 2025-10-27 12:15:00 | 296.60 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-10-23 10:30:00 | 298.00 | 2025-10-27 12:15:00 | 296.60 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-10-23 14:45:00 | 298.00 | 2025-10-27 12:15:00 | 296.60 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-10-27 10:15:00 | 298.00 | 2025-10-27 12:15:00 | 296.60 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-11 10:00:00 | 311.25 | 2025-11-11 14:15:00 | 318.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-11-11 10:45:00 | 311.50 | 2025-11-11 14:15:00 | 318.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-11-14 14:45:00 | 308.40 | 2025-11-17 09:15:00 | 313.20 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-14 15:15:00 | 308.05 | 2025-11-17 09:15:00 | 313.20 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-12 11:30:00 | 348.40 | 2025-12-15 15:15:00 | 348.30 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-12 13:15:00 | 347.25 | 2025-12-15 15:15:00 | 348.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-12 14:00:00 | 347.75 | 2025-12-15 15:15:00 | 348.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-15 15:15:00 | 348.30 | 2025-12-15 15:15:00 | 348.30 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-01-14 12:15:00 | 359.45 | 2026-01-16 09:15:00 | 354.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-22 12:00:00 | 356.00 | 2026-01-22 12:15:00 | 360.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-28 09:45:00 | 370.70 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-01-28 11:00:00 | 370.35 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-28 11:45:00 | 370.10 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-01-28 13:00:00 | 369.80 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-01-29 15:00:00 | 374.55 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2026-01-30 13:45:00 | 376.70 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2026-01-30 14:45:00 | 374.20 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2026-02-01 09:30:00 | 374.50 | 2026-02-01 13:15:00 | 358.05 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2026-02-03 12:30:00 | 361.20 | 2026-02-04 09:15:00 | 366.85 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-03 13:15:00 | 361.00 | 2026-02-04 09:15:00 | 366.85 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-03 13:45:00 | 360.70 | 2026-02-04 09:15:00 | 366.85 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-02-06 11:15:00 | 377.80 | 2026-02-12 09:15:00 | 378.90 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-02-06 14:30:00 | 376.55 | 2026-02-12 09:15:00 | 378.90 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2026-02-09 09:15:00 | 377.65 | 2026-02-12 09:15:00 | 378.90 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2026-03-09 09:15:00 | 346.65 | 2026-03-10 15:15:00 | 360.40 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-03-10 12:30:00 | 361.70 | 2026-03-10 15:15:00 | 360.40 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2026-03-10 14:45:00 | 361.80 | 2026-03-10 15:15:00 | 360.40 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2026-04-01 14:30:00 | 290.85 | 2026-04-02 12:15:00 | 276.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:30:00 | 290.85 | 2026-04-02 14:15:00 | 285.50 | STOP_HIT | 0.50 | 1.84% |
| BUY | retest2 | 2026-04-13 14:45:00 | 298.00 | 2026-04-16 12:15:00 | 294.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-04-15 09:15:00 | 304.20 | 2026-04-16 12:15:00 | 294.30 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-04-16 11:00:00 | 298.05 | 2026-04-16 12:15:00 | 294.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-04-16 11:30:00 | 298.15 | 2026-04-16 12:15:00 | 294.30 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-04-17 15:15:00 | 300.00 | 2026-04-22 14:15:00 | 300.10 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2026-04-20 10:00:00 | 301.05 | 2026-04-22 14:15:00 | 300.10 | STOP_HIT | 1.00 | -0.32% |
