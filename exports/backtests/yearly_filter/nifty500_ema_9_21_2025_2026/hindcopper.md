# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 568.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 63 |
| ALERT1 | 46 |
| ALERT2 | 44 |
| ALERT2_SKIP | 22 |
| ALERT3 | 108 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 45 |
| PARTIAL | 5 |
| TARGET_HIT | 9 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 31
- **Target hits / Stop hits / Partials:** 9 / 40 / 5
- **Avg / median % per leg:** 1.91% / -0.68%
- **Sum % (uncompounded):** 103.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 8 | 17 | 0 | 2.91% | 72.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.68% | -6.7% |
| BUY @ 3rd Alert (retest2) | 21 | 12 | 57.1% | 8 | 13 | 0 | 3.78% | 79.5% |
| SELL (all) | 29 | 11 | 37.9% | 1 | 23 | 5 | 1.05% | 30.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 11 | 37.9% | 1 | 23 | 5 | 1.05% | 30.3% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.68% | -6.7% |
| retest2 (combined) | 50 | 23 | 46.0% | 9 | 36 | 5 | 2.20% | 109.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 217.74 | 209.67 | 208.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 220.89 | 213.13 | 210.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 221.68 | 222.05 | 218.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 221.68 | 222.05 | 218.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 227.75 | 229.09 | 226.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 230.85 | 229.09 | 226.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 229.16 | 229.10 | 226.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 226.11 | 227.89 | 227.15 | SL hit (close<static) qty=1.00 sl=226.61 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 224.90 | 227.96 | 227.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 224.49 | 227.04 | 227.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 227.05 | 227.04 | 227.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 227.05 | 227.04 | 227.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 227.84 | 227.20 | 227.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 227.84 | 227.20 | 227.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 229.30 | 227.62 | 227.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 229.30 | 227.62 | 227.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 228.90 | 227.88 | 227.77 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 226.51 | 227.60 | 227.65 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 228.40 | 227.71 | 227.69 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 226.01 | 227.37 | 227.54 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 229.70 | 227.53 | 227.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 234.71 | 228.97 | 228.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 244.55 | 244.89 | 241.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:15:00 | 256.23 | 244.89 | 241.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 248.40 | 250.18 | 248.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 248.40 | 250.18 | 248.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 248.78 | 249.90 | 248.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 247.72 | 249.90 | 248.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 247.95 | 249.51 | 248.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 247.95 | 249.51 | 248.24 | SL hit (close<ema400) qty=1.00 sl=248.24 alert=retest1 |

### Cycle 8 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 246.73 | 247.84 | 247.86 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 250.62 | 248.39 | 248.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 251.99 | 249.11 | 248.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 249.40 | 250.69 | 249.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 249.40 | 250.69 | 249.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 249.35 | 250.42 | 249.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:30:00 | 249.00 | 250.42 | 249.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 248.99 | 249.76 | 249.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 249.67 | 249.76 | 249.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 254.80 | 259.79 | 260.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 254.80 | 259.79 | 260.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 11:15:00 | 254.50 | 256.48 | 258.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 256.64 | 254.17 | 255.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 256.64 | 254.17 | 255.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 261.51 | 255.63 | 256.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 261.51 | 255.63 | 256.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 263.00 | 257.11 | 256.99 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 253.05 | 257.00 | 257.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 249.46 | 253.26 | 254.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 249.03 | 247.49 | 250.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 249.03 | 247.49 | 250.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 247.99 | 247.76 | 249.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 248.40 | 248.08 | 249.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 249.89 | 248.44 | 249.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 249.89 | 248.44 | 249.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 250.45 | 248.84 | 249.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 250.45 | 248.84 | 249.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 252.80 | 249.63 | 249.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 252.81 | 249.63 | 249.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 252.18 | 250.14 | 249.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 258.27 | 252.14 | 250.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 276.25 | 278.47 | 274.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 276.85 | 278.47 | 274.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 281.90 | 282.80 | 280.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 281.60 | 282.80 | 280.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 281.00 | 282.14 | 281.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 281.40 | 282.14 | 281.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 280.00 | 281.71 | 280.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 280.95 | 281.71 | 280.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 280.80 | 281.53 | 280.96 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 277.45 | 280.33 | 280.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 275.95 | 279.45 | 280.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 275.00 | 274.10 | 275.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 13:00:00 | 275.00 | 274.10 | 275.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 264.35 | 272.01 | 274.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 263.65 | 269.54 | 272.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 263.55 | 267.52 | 271.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 263.55 | 265.78 | 269.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:45:00 | 263.20 | 265.31 | 268.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 265.40 | 265.10 | 267.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 267.40 | 265.10 | 267.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 264.30 | 263.46 | 264.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 264.50 | 263.46 | 264.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 264.50 | 263.67 | 264.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:15:00 | 264.60 | 263.67 | 264.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 263.50 | 263.63 | 264.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 264.45 | 263.63 | 264.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 261.90 | 263.19 | 264.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:30:00 | 261.05 | 262.42 | 263.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 260.20 | 262.56 | 262.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 265.35 | 263.28 | 263.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 265.35 | 263.28 | 263.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 13:15:00 | 267.10 | 264.83 | 264.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 273.85 | 274.68 | 272.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 273.85 | 274.68 | 272.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 272.50 | 274.24 | 272.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 272.50 | 274.24 | 272.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 271.55 | 273.70 | 272.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 278.00 | 273.70 | 272.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 266.95 | 272.85 | 273.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 266.95 | 272.85 | 273.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 264.90 | 271.26 | 272.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 254.80 | 253.57 | 258.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 254.80 | 253.57 | 258.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 259.10 | 255.38 | 258.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 259.10 | 255.38 | 258.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 258.50 | 256.00 | 258.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 259.65 | 256.00 | 258.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 258.40 | 256.48 | 258.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 258.40 | 256.48 | 258.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 258.20 | 256.83 | 258.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 258.05 | 256.83 | 258.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 256.65 | 256.79 | 258.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 254.20 | 256.84 | 258.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 261.90 | 258.08 | 258.48 | SL hit (close>static) qty=1.00 sl=260.80 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 245.49 | 243.95 | 243.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 14:15:00 | 245.64 | 244.32 | 244.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 243.97 | 244.25 | 244.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 244.48 | 244.25 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 241.96 | 243.79 | 243.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 240.46 | 243.12 | 243.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 12:15:00 | 237.60 | 237.50 | 239.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 13:00:00 | 237.60 | 237.50 | 239.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 237.35 | 237.52 | 238.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 238.40 | 237.52 | 238.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 237.80 | 237.62 | 238.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 237.80 | 237.62 | 238.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 236.60 | 237.42 | 238.45 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 242.66 | 239.39 | 239.06 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 240.00 | 240.90 | 240.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 239.11 | 240.43 | 240.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 238.77 | 238.24 | 239.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 238.77 | 238.24 | 239.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 238.91 | 238.38 | 239.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 237.17 | 238.45 | 239.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 232.50 | 231.48 | 231.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 232.50 | 231.48 | 231.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 234.65 | 232.12 | 231.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 245.76 | 245.92 | 242.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 246.45 | 245.92 | 242.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 242.74 | 244.89 | 242.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 242.74 | 244.89 | 242.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 243.35 | 244.58 | 242.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 243.45 | 244.58 | 242.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 241.59 | 243.98 | 242.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 241.59 | 243.98 | 242.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 242.29 | 243.65 | 242.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 244.22 | 242.95 | 242.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 245.14 | 244.70 | 243.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 243.60 | 243.86 | 243.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 11:15:00 | 267.96 | 256.55 | 251.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 279.60 | 281.37 | 281.60 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 283.99 | 281.98 | 281.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 301.56 | 285.89 | 283.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 308.10 | 308.46 | 303.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:15:00 | 316.18 | 308.46 | 303.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 316.76 | 320.80 | 315.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 311.53 | 318.14 | 315.36 | SL hit (close<ema400) qty=1.00 sl=315.36 alert=retest1 |

### Cycle 24 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 332.00 | 335.83 | 336.04 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 339.15 | 336.49 | 336.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 10:15:00 | 339.40 | 337.07 | 336.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 344.95 | 355.09 | 349.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 344.95 | 355.09 | 349.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 344.50 | 352.97 | 349.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 343.20 | 352.97 | 349.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 342.70 | 346.64 | 347.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 337.00 | 340.97 | 343.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 335.20 | 334.02 | 338.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 335.20 | 334.02 | 338.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 341.10 | 335.44 | 338.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 341.10 | 335.44 | 338.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 341.40 | 336.63 | 338.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 341.30 | 336.63 | 338.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 341.50 | 339.57 | 339.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 342.45 | 340.47 | 340.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 341.60 | 342.00 | 341.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 346.00 | 342.00 | 341.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 344.30 | 342.46 | 341.36 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 337.20 | 340.55 | 340.94 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 344.50 | 341.42 | 341.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 346.90 | 342.52 | 341.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 334.40 | 341.28 | 341.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 334.40 | 341.28 | 341.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 332.00 | 339.43 | 340.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 330.60 | 335.39 | 338.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 342.70 | 335.91 | 337.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 343.25 | 335.91 | 337.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 339.60 | 336.65 | 337.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 340.65 | 336.65 | 337.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 342.40 | 338.69 | 338.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 343.75 | 340.42 | 339.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 346.30 | 346.44 | 344.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 346.30 | 346.44 | 344.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 344.25 | 345.97 | 344.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 344.30 | 345.97 | 344.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 345.50 | 345.88 | 344.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 352.15 | 344.29 | 343.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 347.25 | 348.73 | 347.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 341.85 | 346.58 | 347.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 341.85 | 346.58 | 347.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 340.70 | 343.62 | 345.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 343.50 | 342.30 | 343.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 340.05 | 342.30 | 343.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:15:00 | 323.05 | 331.33 | 336.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 326.35 | 325.64 | 330.60 | SL hit (close>ema200) qty=0.50 sl=325.64 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 341.50 | 333.48 | 332.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 344.00 | 338.09 | 335.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 343.70 | 348.53 | 343.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 343.70 | 348.53 | 343.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 342.65 | 347.36 | 343.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 342.65 | 347.36 | 343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 343.25 | 346.54 | 343.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 345.75 | 343.45 | 342.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 340.15 | 342.32 | 342.30 | SL hit (close<static) qty=1.00 sl=340.85 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 337.65 | 341.39 | 341.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 334.40 | 339.41 | 340.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 337.25 | 335.92 | 337.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 337.55 | 335.92 | 337.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 339.85 | 336.71 | 337.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 339.85 | 336.71 | 337.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 339.10 | 337.19 | 337.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 337.65 | 337.49 | 338.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 337.45 | 337.49 | 338.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 320.77 | 325.28 | 328.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 320.58 | 325.28 | 328.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 322.00 | 314.60 | 317.85 | SL hit (close>ema200) qty=0.50 sl=314.60 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 321.45 | 319.21 | 319.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 325.05 | 320.49 | 319.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 326.80 | 327.06 | 324.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 326.80 | 327.06 | 324.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 324.90 | 326.70 | 325.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 324.20 | 326.70 | 325.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 325.40 | 326.44 | 325.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 328.20 | 327.53 | 325.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 336.30 | 327.76 | 326.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-04 10:15:00 | 361.02 | 344.62 | 340.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 358.90 | 362.26 | 362.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 356.45 | 361.26 | 361.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 369.50 | 359.73 | 360.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 369.60 | 359.73 | 360.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 376.60 | 363.10 | 361.81 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 369.40 | 372.00 | 372.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 366.00 | 370.11 | 371.08 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 379.50 | 371.24 | 371.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 10:15:00 | 384.80 | 373.95 | 372.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 381.85 | 382.28 | 378.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:45:00 | 381.95 | 382.28 | 378.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 555.90 | 561.15 | 553.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 554.10 | 561.15 | 553.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 554.00 | 559.18 | 553.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 554.00 | 559.18 | 553.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 556.15 | 558.57 | 553.97 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 532.20 | 550.74 | 551.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 523.50 | 536.10 | 543.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 530.45 | 525.17 | 531.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:15:00 | 538.00 | 525.17 | 531.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 537.00 | 527.53 | 531.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 540.20 | 527.53 | 531.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 535.50 | 529.13 | 532.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:15:00 | 539.25 | 529.13 | 532.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 541.20 | 531.54 | 532.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 539.70 | 531.54 | 532.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 546.20 | 535.65 | 534.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 547.20 | 537.96 | 535.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 539.40 | 541.86 | 538.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 539.40 | 541.86 | 538.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 539.35 | 541.51 | 539.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 552.25 | 541.51 | 539.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 548.35 | 555.48 | 555.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 548.35 | 555.48 | 555.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 540.95 | 551.49 | 553.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 10:15:00 | 543.25 | 540.55 | 545.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 11:00:00 | 543.25 | 540.55 | 545.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 549.70 | 542.38 | 546.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 549.70 | 542.38 | 546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 551.20 | 544.15 | 546.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 551.20 | 544.15 | 546.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 557.00 | 547.41 | 547.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 555.95 | 547.41 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 558.00 | 549.52 | 548.62 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 09:15:00 | 531.00 | 545.82 | 547.02 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 12:15:00 | 546.25 | 543.33 | 543.10 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 540.85 | 542.84 | 542.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 535.00 | 541.27 | 542.17 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 551.25 | 542.90 | 542.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 561.60 | 550.68 | 546.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 708.55 | 722.05 | 671.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 687.90 | 701.15 | 679.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 687.90 | 701.15 | 679.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 674.00 | 701.15 | 679.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 675.10 | 695.94 | 678.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 571.60 | 695.94 | 678.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 604.65 | 677.68 | 672.03 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 614.50 | 665.04 | 666.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 593.20 | 650.68 | 660.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 610.85 | 606.37 | 623.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 610.85 | 606.37 | 623.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 614.55 | 608.58 | 621.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 593.00 | 617.74 | 620.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:00:00 | 605.70 | 605.50 | 611.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 591.65 | 606.42 | 611.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 606.35 | 600.05 | 599.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 606.35 | 600.05 | 599.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 617.75 | 603.59 | 600.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 602.45 | 615.34 | 610.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 604.60 | 615.34 | 610.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 601.85 | 612.64 | 609.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 601.85 | 612.64 | 609.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 599.00 | 606.06 | 606.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 591.40 | 603.13 | 605.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 567.25 | 559.79 | 571.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 567.25 | 559.79 | 571.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 567.55 | 561.34 | 571.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 565.20 | 562.72 | 570.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 575.40 | 565.26 | 571.27 | SL hit (close>static) qty=1.00 sl=572.50 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 576.90 | 567.05 | 566.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 581.95 | 570.03 | 567.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 580.80 | 582.06 | 577.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 14:45:00 | 583.85 | 582.32 | 577.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 585.15 | 582.46 | 578.14 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 586.15 | 583.20 | 578.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 578.60 | 582.10 | 579.11 | SL hit (close<ema400) qty=1.00 sl=579.11 alert=retest1 |

### Cycle 52 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 570.00 | 576.84 | 577.45 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 584.20 | 578.32 | 578.06 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 574.80 | 577.61 | 577.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 573.60 | 576.81 | 577.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 577.00 | 575.64 | 576.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 560.80 | 575.64 | 576.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 532.76 | 541.71 | 546.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 504.72 | 525.18 | 534.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 489.20 | 482.88 | 482.33 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 463.30 | 478.96 | 480.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 457.45 | 474.66 | 478.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 459.95 | 455.80 | 463.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 459.75 | 455.80 | 463.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 467.00 | 458.04 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 467.00 | 458.04 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 464.40 | 459.31 | 463.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 463.65 | 459.31 | 463.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 463.00 | 460.05 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 483.45 | 460.05 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 482.05 | 464.45 | 465.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:15:00 | 484.25 | 464.45 | 465.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 484.40 | 468.44 | 467.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 13:15:00 | 489.40 | 478.37 | 474.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 471.60 | 481.31 | 477.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 471.60 | 481.31 | 477.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 465.85 | 478.22 | 476.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 465.85 | 478.22 | 476.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 463.60 | 473.87 | 474.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 458.75 | 470.84 | 473.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 482.35 | 468.26 | 470.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 482.35 | 468.26 | 470.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 498.40 | 474.29 | 473.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 510.15 | 485.42 | 478.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 474.00 | 487.83 | 482.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 475.05 | 487.83 | 482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 471.25 | 484.52 | 481.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 470.35 | 484.52 | 481.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 502.70 | 491.56 | 486.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:45:00 | 508.25 | 501.56 | 494.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 508.30 | 503.51 | 496.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 508.40 | 504.72 | 498.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 522.95 | 504.61 | 499.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 519.95 | 527.25 | 524.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 523.35 | 527.25 | 524.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 559.08 | 534.25 | 529.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 554.25 | 558.24 | 558.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 552.90 | 556.70 | 557.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 556.60 | 555.96 | 557.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 557.25 | 555.96 | 557.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 555.25 | 555.82 | 556.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 552.00 | 556.52 | 556.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 552.80 | 555.78 | 556.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:00:00 | 552.60 | 554.23 | 555.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 558.25 | 551.95 | 551.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 558.25 | 551.95 | 551.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 565.60 | 557.22 | 554.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 558.50 | 558.88 | 555.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:30:00 | 560.85 | 558.88 | 555.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 557.90 | 558.07 | 556.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 550.25 | 558.07 | 556.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 552.80 | 557.02 | 555.86 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 541.00 | 553.32 | 554.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 540.30 | 546.50 | 550.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 545.50 | 535.80 | 538.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 542.20 | 535.80 | 538.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 547.00 | 540.36 | 540.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 547.00 | 540.36 | 540.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 548.70 | 542.03 | 540.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 563.80 | 566.09 | 557.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:30:00 | 573.75 | 566.80 | 559.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 09:15:00 | 230.85 | 2025-05-16 14:15:00 | 226.11 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-05-16 10:00:00 | 229.16 | 2025-05-16 14:15:00 | 226.11 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-19 09:30:00 | 229.05 | 2025-05-19 15:15:00 | 226.15 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-20 10:00:00 | 230.40 | 2025-05-20 14:15:00 | 224.90 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2025-05-28 09:15:00 | 256.23 | 2025-05-30 11:15:00 | 247.95 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-06-04 09:15:00 | 249.67 | 2025-06-12 13:15:00 | 254.80 | STOP_HIT | 1.00 | 2.05% |
| SELL | retest2 | 2025-07-09 12:00:00 | 263.65 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-09 13:30:00 | 263.55 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-10 10:45:00 | 263.55 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-10 11:45:00 | 263.20 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-16 09:30:00 | 261.05 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-16 14:45:00 | 260.20 | 2025-07-17 09:15:00 | 265.35 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-23 09:15:00 | 278.00 | 2025-07-25 09:15:00 | 266.95 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-07-30 13:15:00 | 254.20 | 2025-07-30 14:15:00 | 261.90 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-07-31 09:30:00 | 253.20 | 2025-08-01 11:15:00 | 240.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:30:00 | 253.20 | 2025-08-04 12:15:00 | 242.00 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-08-25 12:15:00 | 237.17 | 2025-09-01 15:15:00 | 232.50 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-09-05 13:45:00 | 244.22 | 2025-09-12 11:15:00 | 267.96 | TARGET_HIT | 1.00 | 9.72% |
| BUY | retest2 | 2025-09-08 15:00:00 | 245.14 | 2025-09-12 12:15:00 | 268.64 | TARGET_HIT | 1.00 | 9.59% |
| BUY | retest2 | 2025-09-09 15:00:00 | 243.60 | 2025-09-12 12:15:00 | 269.65 | TARGET_HIT | 1.00 | 10.70% |
| BUY | retest1 | 2025-09-25 09:15:00 | 316.18 | 2025-09-26 13:15:00 | 311.53 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-29 09:15:00 | 321.26 | 2025-10-07 15:15:00 | 332.00 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2025-10-29 09:15:00 | 352.15 | 2025-10-31 10:15:00 | 341.85 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-10-30 11:45:00 | 347.25 | 2025-10-31 10:15:00 | 341.85 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-04 09:15:00 | 340.05 | 2025-11-06 10:15:00 | 323.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 340.05 | 2025-11-07 09:15:00 | 326.35 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-11-10 09:45:00 | 339.55 | 2025-11-10 10:15:00 | 341.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-13 09:30:00 | 345.75 | 2025-11-13 13:15:00 | 340.15 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-17 14:30:00 | 337.65 | 2025-11-21 09:15:00 | 320.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 337.45 | 2025-11-21 09:15:00 | 320.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:30:00 | 337.65 | 2025-11-25 09:15:00 | 322.00 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-11-17 15:15:00 | 337.45 | 2025-11-25 09:15:00 | 322.00 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest2 | 2025-11-28 10:45:00 | 328.20 | 2025-12-04 10:15:00 | 361.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-01 09:15:00 | 336.30 | 2025-12-05 09:15:00 | 369.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-14 09:15:00 | 552.25 | 2026-01-19 14:15:00 | 548.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-05 09:15:00 | 593.00 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-02-05 15:00:00 | 605.70 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-02-06 09:15:00 | 591.65 | 2026-02-11 15:15:00 | 606.35 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-02-18 12:15:00 | 565.20 | 2026-02-18 12:15:00 | 575.40 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-19 15:00:00 | 562.85 | 2026-02-23 09:15:00 | 574.30 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-23 10:45:00 | 565.50 | 2026-02-25 09:15:00 | 574.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2026-02-26 14:45:00 | 583.85 | 2026-02-27 11:15:00 | 578.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2026-02-27 09:15:00 | 585.15 | 2026-02-27 11:15:00 | 578.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-04 09:15:00 | 560.80 | 2026-03-12 09:15:00 | 532.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 560.80 | 2026-03-13 09:15:00 | 504.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-07 09:45:00 | 508.25 | 2026-04-15 09:15:00 | 559.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:45:00 | 508.30 | 2026-04-15 09:15:00 | 559.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 13:30:00 | 508.40 | 2026-04-15 09:15:00 | 559.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 522.95 | 2026-04-21 12:15:00 | 554.25 | STOP_HIT | 1.00 | 5.99% |
| BUY | retest2 | 2026-04-13 10:15:00 | 523.35 | 2026-04-21 12:15:00 | 554.25 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2026-04-23 09:15:00 | 552.00 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-04-23 10:00:00 | 552.80 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-23 12:00:00 | 552.60 | 2026-04-27 12:15:00 | 558.25 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-05-06 10:15:00 | 542.20 | 2026-05-06 12:15:00 | 547.00 | STOP_HIT | 1.00 | -0.89% |
