# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 427.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 125 |
| ALERT1 | 87 |
| ALERT2 | 87 |
| ALERT2_SKIP | 39 |
| ALERT3 | 212 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 114 |
| PARTIAL | 14 |
| TARGET_HIT | 15 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 81
- **Target hits / Stop hits / Partials:** 15 / 108 / 14
- **Avg / median % per leg:** 0.90% / -1.05%
- **Sum % (uncompounded):** 123.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 17 | 29.3% | 10 | 48 | 0 | 0.68% | 39.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.81% | -7.2% |
| BUY @ 3rd Alert (retest2) | 54 | 17 | 31.5% | 10 | 44 | 0 | 0.87% | 46.8% |
| SELL (all) | 79 | 39 | 49.4% | 5 | 60 | 14 | 1.06% | 83.8% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.24% | -1.4% |
| SELL @ 3rd Alert (retest2) | 73 | 37 | 50.7% | 5 | 55 | 13 | 1.17% | 85.2% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.87% | -8.7% |
| retest2 (combined) | 127 | 54 | 42.5% | 15 | 99 | 13 | 1.04% | 132.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 218.85 | 214.90 | 214.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 220.35 | 215.99 | 215.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 227.40 | 228.88 | 226.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 227.40 | 228.88 | 226.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 224.55 | 228.02 | 226.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 224.00 | 228.02 | 226.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 224.90 | 227.39 | 225.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 223.80 | 227.39 | 225.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 222.90 | 226.50 | 225.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 224.00 | 226.50 | 225.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 217.05 | 224.21 | 224.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 10:15:00 | 214.95 | 222.36 | 223.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 12:15:00 | 213.80 | 213.38 | 217.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 12:45:00 | 213.85 | 213.38 | 217.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 206.65 | 206.02 | 208.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:30:00 | 205.70 | 206.45 | 207.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 209.75 | 208.08 | 208.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 209.75 | 208.08 | 208.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 206.50 | 207.98 | 208.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 205.60 | 207.26 | 207.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 204.55 | 204.43 | 205.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 204.50 | 204.43 | 205.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 209.25 | 205.40 | 205.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 209.25 | 205.40 | 205.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 205.45 | 205.41 | 205.88 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 211.80 | 206.81 | 206.44 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 200.90 | 207.30 | 207.80 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 213.05 | 207.46 | 207.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 219.15 | 212.35 | 209.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 245.78 | 246.47 | 238.48 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 12:45:00 | 250.80 | 247.15 | 240.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 250.86 | 247.77 | 242.70 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:30:00 | 248.90 | 248.55 | 244.40 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 12:30:00 | 248.70 | 248.78 | 244.88 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 245.30 | 247.32 | 245.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 245.30 | 247.32 | 245.38 | SL hit (close<ema400) qty=1.00 sl=245.38 alert=retest1 |

### Cycle 8 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 264.36 | 267.00 | 267.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 262.50 | 266.10 | 266.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 261.04 | 260.31 | 262.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:30:00 | 261.53 | 260.31 | 262.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 263.86 | 261.02 | 262.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 263.86 | 261.02 | 262.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 262.22 | 261.26 | 262.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:00:00 | 261.00 | 261.19 | 262.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:45:00 | 259.85 | 260.08 | 260.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 15:15:00 | 261.50 | 260.56 | 260.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 261.50 | 260.56 | 260.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 263.90 | 262.42 | 261.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 257.40 | 261.67 | 261.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 257.40 | 261.67 | 261.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 257.40 | 261.67 | 261.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 257.40 | 261.67 | 261.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 10:15:00 | 258.30 | 260.99 | 261.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 254.75 | 258.32 | 259.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 12:15:00 | 255.30 | 253.18 | 255.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 12:15:00 | 255.30 | 253.18 | 255.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 255.30 | 253.18 | 255.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 255.50 | 253.18 | 255.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 259.30 | 254.40 | 255.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 259.00 | 254.40 | 255.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 259.15 | 255.35 | 255.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 259.15 | 255.35 | 255.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 263.15 | 257.53 | 256.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 284.00 | 263.92 | 260.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 274.20 | 274.34 | 268.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:45:00 | 276.20 | 274.26 | 268.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 273.00 | 274.36 | 271.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 271.20 | 274.36 | 271.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 293.40 | 292.97 | 291.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 286.40 | 292.97 | 291.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 281.70 | 290.72 | 290.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 281.70 | 290.72 | 290.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 281.95 | 288.97 | 289.67 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 292.80 | 287.68 | 287.22 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 286.70 | 289.78 | 290.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 285.30 | 287.78 | 288.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 11:15:00 | 286.65 | 286.43 | 287.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 286.65 | 286.43 | 287.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 286.65 | 286.43 | 287.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:30:00 | 287.50 | 286.43 | 287.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 292.10 | 287.56 | 288.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:45:00 | 295.95 | 287.56 | 288.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 294.20 | 288.89 | 288.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 13:15:00 | 298.40 | 292.85 | 291.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 296.25 | 304.78 | 301.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 296.25 | 304.78 | 301.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 296.25 | 304.78 | 301.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 296.50 | 304.78 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 292.50 | 302.32 | 300.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 292.50 | 302.32 | 300.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 295.50 | 299.04 | 299.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 293.00 | 297.83 | 298.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 298.60 | 297.99 | 298.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 298.60 | 297.99 | 298.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 298.60 | 297.99 | 298.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 298.35 | 297.99 | 298.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 295.70 | 297.53 | 298.37 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 309.10 | 299.20 | 298.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 315.70 | 307.35 | 303.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 305.80 | 308.40 | 305.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 305.80 | 308.40 | 305.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 307.90 | 308.30 | 305.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 310.05 | 308.30 | 305.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:45:00 | 308.45 | 308.61 | 306.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:00:00 | 310.05 | 308.96 | 307.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 310.05 | 308.76 | 307.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 306.95 | 308.40 | 307.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 312.30 | 307.72 | 307.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:45:00 | 312.35 | 308.83 | 307.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 305.30 | 308.10 | 307.94 | SL hit (close<static) qty=1.00 sl=305.50 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 303.50 | 307.18 | 307.53 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 310.00 | 307.80 | 307.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 13:15:00 | 313.25 | 308.89 | 308.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 335.50 | 336.31 | 330.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 335.50 | 336.31 | 330.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 346.40 | 348.01 | 345.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 351.95 | 348.01 | 345.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-02 09:15:00 | 387.15 | 378.76 | 374.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 385.10 | 400.47 | 400.71 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 401.95 | 397.63 | 397.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 403.55 | 398.81 | 397.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 405.25 | 407.21 | 404.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 405.25 | 407.21 | 404.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 405.25 | 407.21 | 404.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:15:00 | 402.35 | 407.21 | 404.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 402.50 | 406.27 | 404.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 406.90 | 403.85 | 403.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:00:00 | 406.40 | 405.06 | 404.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 406.10 | 405.47 | 404.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 14:15:00 | 406.20 | 405.47 | 404.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-16 10:15:00 | 447.04 | 418.52 | 411.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 419.35 | 426.50 | 426.69 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 426.40 | 425.38 | 425.35 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 13:15:00 | 422.50 | 425.06 | 425.22 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 436.30 | 427.31 | 426.23 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 423.15 | 429.71 | 430.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 420.20 | 427.81 | 429.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 403.70 | 401.19 | 407.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 403.70 | 401.19 | 407.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 403.70 | 401.19 | 407.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:00:00 | 398.10 | 401.71 | 405.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 400.90 | 401.45 | 404.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 378.19 | 392.04 | 395.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 380.85 | 392.04 | 395.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 12:15:00 | 390.10 | 387.15 | 392.16 | SL hit (close>ema200) qty=0.50 sl=387.15 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 396.85 | 391.63 | 391.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 403.00 | 395.30 | 393.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 399.15 | 400.53 | 397.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 399.15 | 400.53 | 397.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 397.95 | 400.06 | 398.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 397.25 | 400.06 | 398.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 399.45 | 399.93 | 398.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 400.30 | 399.93 | 398.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:30:00 | 400.30 | 400.56 | 399.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:30:00 | 402.75 | 401.02 | 399.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 393.90 | 400.31 | 399.94 | SL hit (close<static) qty=1.00 sl=397.45 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 395.65 | 399.38 | 399.55 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 400.60 | 399.61 | 399.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 404.00 | 400.47 | 399.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 414.30 | 415.43 | 410.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 414.30 | 415.43 | 410.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 417.95 | 418.78 | 414.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 414.25 | 418.78 | 414.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 415.45 | 418.11 | 414.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 421.10 | 417.47 | 415.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:45:00 | 420.95 | 418.96 | 416.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 15:15:00 | 410.50 | 415.36 | 415.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 410.50 | 415.36 | 415.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 407.55 | 413.80 | 415.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 13:15:00 | 412.65 | 411.25 | 413.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 14:00:00 | 412.65 | 411.25 | 413.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 397.80 | 408.18 | 411.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 388.95 | 407.76 | 408.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-24 14:15:00 | 350.06 | 396.10 | 403.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 379.85 | 374.83 | 374.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 380.20 | 376.99 | 376.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 388.95 | 389.57 | 384.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 11:00:00 | 388.95 | 389.57 | 384.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 382.60 | 392.68 | 389.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 382.60 | 392.68 | 389.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 379.75 | 390.10 | 388.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 380.00 | 390.10 | 388.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 379.95 | 386.40 | 386.80 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 395.35 | 387.41 | 386.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 397.30 | 391.15 | 388.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 392.50 | 393.24 | 390.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 392.50 | 393.24 | 390.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 394.60 | 393.51 | 390.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 390.15 | 393.51 | 390.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 391.20 | 393.17 | 391.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:45:00 | 390.40 | 393.17 | 391.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 391.60 | 392.85 | 391.25 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 383.40 | 389.61 | 390.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 382.00 | 388.08 | 389.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 376.05 | 374.61 | 379.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 376.05 | 374.61 | 379.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 376.05 | 374.61 | 379.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 365.10 | 373.09 | 377.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 346.85 | 357.54 | 362.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 353.75 | 349.05 | 354.65 | SL hit (close>ema200) qty=0.50 sl=349.05 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 353.05 | 349.30 | 349.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 355.55 | 352.89 | 351.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 385.50 | 387.75 | 379.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:45:00 | 386.95 | 387.75 | 379.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 434.25 | 438.46 | 435.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 434.40 | 438.46 | 435.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 428.70 | 436.51 | 434.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 428.70 | 436.51 | 434.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 429.30 | 435.07 | 434.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:00:00 | 429.30 | 435.07 | 434.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 429.90 | 433.45 | 433.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 427.00 | 432.16 | 432.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 433.40 | 431.69 | 432.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 433.40 | 431.69 | 432.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 433.40 | 431.69 | 432.57 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 434.95 | 433.19 | 433.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 15:15:00 | 437.00 | 434.45 | 433.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 14:15:00 | 429.65 | 435.44 | 434.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 429.65 | 435.44 | 434.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 429.65 | 435.44 | 434.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 429.65 | 435.44 | 434.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 431.80 | 434.71 | 434.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 426.25 | 434.71 | 434.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 424.55 | 432.68 | 433.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 421.65 | 430.48 | 432.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 425.30 | 423.28 | 427.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 425.30 | 423.28 | 427.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 425.30 | 423.28 | 427.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 418.80 | 424.43 | 426.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 415.15 | 422.71 | 424.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:15:00 | 418.85 | 421.12 | 422.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 418.15 | 418.96 | 421.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 400.25 | 411.59 | 416.14 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 397.86 | 403.18 | 406.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 397.91 | 403.18 | 406.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 397.24 | 403.18 | 406.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 406.00 | 402.08 | 404.03 | SL hit (close>ema200) qty=0.50 sl=402.08 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 410.55 | 404.40 | 404.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 415.20 | 406.56 | 405.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 415.20 | 415.47 | 411.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:00:00 | 415.20 | 415.47 | 411.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 412.70 | 415.12 | 412.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 412.70 | 415.12 | 412.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 420.65 | 416.23 | 412.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 428.50 | 417.27 | 413.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 408.05 | 425.57 | 426.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 408.05 | 425.57 | 426.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 406.00 | 414.16 | 420.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 429.05 | 417.14 | 420.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 429.05 | 417.14 | 420.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 429.05 | 417.14 | 420.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 426.20 | 417.14 | 420.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 427.60 | 419.23 | 421.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:15:00 | 429.90 | 419.23 | 421.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 427.75 | 423.26 | 422.97 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 418.45 | 422.52 | 422.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 413.85 | 420.78 | 421.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 15:15:00 | 420.65 | 419.41 | 420.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 09:15:00 | 411.95 | 419.41 | 420.82 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 391.35 | 408.31 | 413.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 405.50 | 403.54 | 407.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 405.50 | 403.54 | 407.93 | SL hit (close>ema200) qty=0.50 sl=403.54 alert=retest1 |

### Cycle 43 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 400.45 | 396.61 | 396.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 404.00 | 398.82 | 397.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 399.50 | 400.66 | 399.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 399.50 | 400.66 | 399.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 399.50 | 400.66 | 399.35 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 395.60 | 398.59 | 398.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 389.90 | 396.61 | 397.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 396.30 | 396.09 | 397.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 11:15:00 | 396.30 | 396.09 | 397.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 396.30 | 396.09 | 397.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 397.30 | 396.09 | 397.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 399.30 | 396.73 | 397.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 399.30 | 396.73 | 397.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 399.25 | 397.24 | 397.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:15:00 | 399.30 | 397.24 | 397.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 392.45 | 396.27 | 397.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 389.05 | 396.27 | 397.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 390.40 | 393.94 | 395.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 397.70 | 391.70 | 391.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 397.70 | 391.70 | 391.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 401.85 | 394.77 | 393.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 400.40 | 401.14 | 397.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 400.40 | 401.14 | 397.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 398.70 | 400.65 | 398.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 398.85 | 400.65 | 398.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 396.90 | 399.90 | 397.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 383.80 | 399.90 | 397.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 376.65 | 395.25 | 396.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 368.90 | 382.71 | 388.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 361.40 | 360.10 | 369.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 362.50 | 360.10 | 369.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 367.65 | 363.43 | 369.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 370.40 | 363.43 | 369.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 370.45 | 364.84 | 369.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:45:00 | 370.55 | 364.84 | 369.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 371.35 | 366.14 | 369.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 372.90 | 366.14 | 369.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 372.75 | 367.46 | 369.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 370.00 | 367.46 | 369.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 370.55 | 369.34 | 370.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 367.10 | 368.94 | 369.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 379.10 | 371.09 | 370.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 379.10 | 371.09 | 370.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 383.85 | 373.64 | 371.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 385.95 | 386.12 | 380.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 12:30:00 | 386.80 | 386.12 | 380.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 379.55 | 384.85 | 381.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 379.55 | 384.85 | 381.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 372.60 | 382.40 | 381.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 372.60 | 382.40 | 381.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 371.15 | 380.15 | 380.16 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 384.45 | 378.76 | 378.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 389.50 | 384.00 | 381.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 392.70 | 396.26 | 392.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 392.70 | 396.26 | 392.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 392.70 | 396.26 | 392.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 392.70 | 396.26 | 392.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 396.75 | 396.36 | 392.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 399.40 | 396.36 | 392.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 399.40 | 398.96 | 396.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:30:00 | 399.00 | 398.57 | 396.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 398.70 | 398.57 | 396.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 394.50 | 397.75 | 396.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:00:00 | 394.50 | 397.75 | 396.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 396.55 | 397.51 | 396.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:30:00 | 393.60 | 397.51 | 396.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 393.65 | 396.74 | 396.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 393.85 | 396.74 | 396.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 393.85 | 396.16 | 395.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 393.75 | 396.16 | 395.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-11 10:15:00 | 387.95 | 394.52 | 395.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 387.95 | 394.52 | 395.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 385.20 | 391.74 | 393.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 385.30 | 385.21 | 389.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 383.10 | 385.21 | 389.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 391.00 | 386.36 | 389.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 391.00 | 386.36 | 389.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 395.00 | 388.09 | 389.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 393.50 | 388.09 | 389.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 388.45 | 388.16 | 389.80 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 403.20 | 391.85 | 391.14 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 385.90 | 391.19 | 391.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 384.15 | 389.79 | 391.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 376.55 | 368.64 | 373.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 376.55 | 368.64 | 373.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 376.55 | 368.64 | 373.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 376.55 | 368.64 | 373.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 382.80 | 371.47 | 374.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 382.30 | 371.47 | 374.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 382.35 | 377.04 | 376.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 385.10 | 378.65 | 377.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 381.45 | 382.55 | 380.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 381.45 | 382.55 | 380.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 382.50 | 383.71 | 381.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 382.50 | 383.71 | 381.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 383.70 | 383.71 | 381.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 388.40 | 383.71 | 381.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 380.00 | 382.96 | 381.39 | SL hit (close<static) qty=1.00 sl=380.15 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 377.60 | 380.68 | 380.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 368.70 | 376.85 | 378.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 331.70 | 330.43 | 340.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 331.70 | 330.43 | 340.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 338.60 | 327.40 | 331.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 338.60 | 327.40 | 331.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 332.00 | 328.32 | 331.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 338.90 | 328.32 | 331.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 331.40 | 328.94 | 331.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 333.15 | 328.94 | 331.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 331.95 | 329.54 | 331.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 331.55 | 329.54 | 331.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 337.85 | 331.20 | 332.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:00:00 | 337.85 | 331.20 | 332.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 338.35 | 332.63 | 333.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:45:00 | 338.50 | 332.63 | 333.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 337.00 | 333.51 | 333.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 350.55 | 336.91 | 334.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 350.45 | 350.67 | 346.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 350.45 | 350.67 | 346.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 352.00 | 352.53 | 348.71 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 345.15 | 348.13 | 348.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 344.15 | 346.22 | 347.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 346.65 | 345.97 | 346.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:00:00 | 346.65 | 345.97 | 346.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 343.85 | 345.55 | 346.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 346.80 | 345.55 | 346.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 346.75 | 339.59 | 341.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:00:00 | 346.75 | 339.59 | 341.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 346.70 | 341.01 | 342.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 347.50 | 341.01 | 342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 345.25 | 342.36 | 342.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 345.25 | 342.36 | 342.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 341.50 | 342.19 | 342.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 346.20 | 342.19 | 342.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 355.00 | 344.54 | 343.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 357.85 | 351.31 | 347.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 10:15:00 | 376.10 | 376.12 | 369.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:00:00 | 376.10 | 376.12 | 369.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 377.30 | 384.74 | 380.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 377.30 | 384.74 | 380.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 377.30 | 383.25 | 380.29 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 370.70 | 377.49 | 378.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 368.45 | 373.65 | 376.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 370.10 | 369.19 | 372.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 370.10 | 369.19 | 372.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 367.50 | 368.85 | 371.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 365.85 | 368.85 | 371.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 375.35 | 370.15 | 372.06 | SL hit (close>static) qty=1.00 sl=374.90 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 387.30 | 375.80 | 374.43 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 362.55 | 375.76 | 376.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 361.80 | 368.72 | 372.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 366.50 | 365.08 | 369.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:45:00 | 367.20 | 365.08 | 369.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 369.75 | 366.60 | 369.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 369.70 | 366.60 | 369.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 371.45 | 367.57 | 369.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 371.45 | 367.57 | 369.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 370.05 | 368.07 | 369.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:45:00 | 369.45 | 368.07 | 369.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 369.90 | 368.43 | 369.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 364.40 | 368.43 | 369.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 327.96 | 352.50 | 359.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 348.80 | 336.57 | 336.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 355.30 | 342.07 | 338.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 356.00 | 356.97 | 351.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:45:00 | 354.90 | 356.97 | 351.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 352.50 | 355.29 | 351.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 351.40 | 355.29 | 351.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 351.15 | 354.46 | 351.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:30:00 | 350.85 | 354.46 | 351.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 351.50 | 353.87 | 351.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 351.85 | 353.87 | 351.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 353.25 | 353.75 | 351.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 357.45 | 354.49 | 352.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:15:00 | 357.80 | 355.44 | 353.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:45:00 | 357.10 | 355.41 | 354.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:45:00 | 357.25 | 355.85 | 354.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 364.90 | 367.87 | 365.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 364.90 | 367.87 | 365.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 364.50 | 367.20 | 365.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 361.85 | 367.20 | 365.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 351.60 | 361.93 | 363.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 351.60 | 361.93 | 363.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 346.50 | 351.59 | 353.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 344.10 | 343.99 | 347.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:00:00 | 344.10 | 343.99 | 347.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 344.35 | 344.67 | 346.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 346.30 | 344.67 | 346.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 344.15 | 344.86 | 346.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 342.40 | 344.04 | 345.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 11:45:00 | 342.85 | 339.81 | 342.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 350.75 | 343.56 | 343.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 350.75 | 343.56 | 343.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 356.50 | 347.41 | 345.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 349.80 | 352.13 | 348.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 349.80 | 352.13 | 348.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 349.80 | 352.13 | 348.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 349.80 | 352.13 | 348.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 346.00 | 350.91 | 348.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 342.50 | 350.91 | 348.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 349.60 | 350.65 | 348.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:45:00 | 351.00 | 350.72 | 349.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 351.50 | 350.87 | 349.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 12:15:00 | 386.10 | 377.49 | 372.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 461.15 | 466.35 | 466.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 460.40 | 464.05 | 465.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 455.45 | 453.30 | 456.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 449.60 | 452.40 | 455.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 427.12 | 436.61 | 442.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 436.30 | 431.74 | 437.53 | SL hit (close>ema200) qty=0.50 sl=431.74 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 439.95 | 425.20 | 424.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 443.75 | 428.91 | 426.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 440.40 | 441.18 | 435.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 440.40 | 441.18 | 435.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 469.85 | 477.70 | 468.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 469.95 | 477.70 | 468.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 478.60 | 477.88 | 469.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 466.30 | 477.88 | 469.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 488.00 | 494.56 | 489.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 487.60 | 494.56 | 489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 486.75 | 493.00 | 489.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 484.50 | 493.00 | 489.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 488.75 | 492.15 | 489.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 490.50 | 490.13 | 488.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 480.10 | 487.66 | 487.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 480.10 | 487.66 | 487.79 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 491.60 | 486.06 | 485.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 495.30 | 489.52 | 487.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 492.35 | 493.62 | 491.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 492.35 | 493.62 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 490.95 | 493.08 | 491.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 490.95 | 493.08 | 491.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 486.75 | 491.82 | 491.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 486.05 | 491.82 | 491.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 485.75 | 490.60 | 490.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 483.55 | 487.69 | 489.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 488.00 | 484.50 | 486.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 491.10 | 484.50 | 486.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 496.50 | 486.90 | 487.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 496.30 | 486.90 | 487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 493.00 | 488.12 | 487.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 497.00 | 492.43 | 490.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 504.25 | 506.26 | 501.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 504.25 | 506.26 | 501.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 506.90 | 506.39 | 502.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 510.65 | 507.31 | 502.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 499.10 | 504.96 | 503.51 | SL hit (close<static) qty=1.00 sl=500.90 alert=retest2 |

### Cycle 70 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 501.60 | 502.44 | 502.55 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 504.00 | 502.74 | 502.66 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 501.85 | 502.56 | 502.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 497.95 | 501.64 | 502.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 490.60 | 489.27 | 492.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 490.60 | 489.27 | 492.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 490.80 | 489.57 | 492.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 486.35 | 489.56 | 492.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:45:00 | 488.85 | 488.42 | 490.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 13:45:00 | 486.70 | 487.57 | 490.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 487.10 | 487.93 | 489.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 488.90 | 488.13 | 489.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 490.00 | 488.13 | 489.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 485.50 | 487.60 | 489.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 483.35 | 487.05 | 489.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:45:00 | 484.00 | 484.41 | 487.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 482.30 | 483.20 | 486.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 496.00 | 485.98 | 486.64 | SL hit (close>static) qty=1.00 sl=493.75 alert=retest2 |

### Cycle 73 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 496.75 | 488.14 | 487.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 500.85 | 490.68 | 488.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 494.65 | 497.58 | 494.66 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 487.40 | 493.13 | 493.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 479.95 | 489.51 | 491.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 481.85 | 481.32 | 485.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 483.40 | 481.32 | 485.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 484.00 | 481.86 | 484.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 484.00 | 481.86 | 484.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 485.00 | 482.86 | 484.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 486.10 | 482.86 | 484.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 485.60 | 483.41 | 484.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 482.70 | 483.41 | 484.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 467.95 | 475.04 | 479.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 464.35 | 472.59 | 477.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 464.40 | 468.62 | 473.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 441.13 | 455.23 | 460.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 441.18 | 455.23 | 460.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 455.95 | 455.38 | 459.75 | SL hit (close>ema200) qty=0.50 sl=455.38 alert=retest2 |

### Cycle 75 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 463.20 | 461.44 | 461.26 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 457.20 | 460.55 | 460.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 454.00 | 458.03 | 459.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 452.30 | 450.14 | 453.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 451.15 | 450.14 | 453.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 448.90 | 448.56 | 451.09 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 454.75 | 452.58 | 452.40 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 450.40 | 452.28 | 452.34 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 455.75 | 453.02 | 452.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 456.60 | 453.91 | 453.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 452.60 | 453.69 | 453.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 452.60 | 453.69 | 453.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 448.75 | 452.70 | 452.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 443.80 | 449.53 | 451.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 436.60 | 432.48 | 437.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 423.10 | 435.57 | 437.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 423.15 | 428.83 | 432.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 424.60 | 427.21 | 430.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 431.65 | 429.39 | 429.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 431.65 | 429.39 | 429.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 436.90 | 430.89 | 429.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 445.20 | 446.44 | 441.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 445.20 | 446.44 | 441.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 441.45 | 445.44 | 441.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 441.45 | 445.44 | 441.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 442.05 | 444.76 | 441.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 444.70 | 442.34 | 441.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 447.40 | 443.77 | 442.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 445.00 | 444.01 | 442.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 440.85 | 443.38 | 442.59 | SL hit (close<static) qty=1.00 sl=441.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 440.00 | 442.26 | 442.42 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 443.90 | 442.59 | 442.56 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 440.20 | 442.12 | 442.36 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 450.20 | 443.46 | 442.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 453.90 | 447.34 | 445.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 460.60 | 461.63 | 457.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 459.05 | 461.63 | 457.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 460.40 | 461.39 | 458.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 465.70 | 460.98 | 460.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 464.95 | 462.41 | 461.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 465.00 | 463.04 | 461.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 459.25 | 466.07 | 466.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 459.25 | 466.07 | 466.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 457.85 | 464.42 | 465.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 13:15:00 | 408.05 | 406.46 | 412.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 408.05 | 406.46 | 412.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 402.75 | 403.33 | 406.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 401.35 | 403.45 | 406.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 399.30 | 403.67 | 405.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 400.85 | 398.01 | 400.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 399.60 | 398.01 | 400.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 397.35 | 397.88 | 400.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 393.55 | 396.72 | 398.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:00:00 | 393.25 | 396.72 | 398.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 392.30 | 393.01 | 395.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 393.00 | 393.01 | 395.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 396.00 | 390.90 | 392.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 396.00 | 390.90 | 392.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 396.55 | 392.03 | 393.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 396.55 | 392.03 | 393.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 400.80 | 393.78 | 393.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 400.80 | 393.78 | 393.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 12:15:00 | 404.70 | 400.50 | 398.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 408.30 | 409.66 | 406.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 10:00:00 | 408.30 | 409.66 | 406.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 405.40 | 408.44 | 406.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 405.30 | 408.44 | 406.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 411.30 | 409.02 | 406.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 413.15 | 409.02 | 406.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 413.15 | 409.84 | 407.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 412.20 | 414.58 | 411.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 416.80 | 415.54 | 412.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 423.30 | 424.91 | 422.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 423.55 | 424.91 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 426.50 | 425.23 | 422.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-27 15:15:00 | 421.90 | 422.96 | 423.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 421.90 | 422.96 | 423.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 420.15 | 422.40 | 422.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 417.85 | 417.56 | 419.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 414.80 | 417.58 | 418.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 419.60 | 417.92 | 418.66 | SL hit (close>static) qty=1.00 sl=419.50 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 15:15:00 | 420.40 | 418.63 | 418.50 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 411.65 | 417.23 | 417.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 410.05 | 415.80 | 417.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 414.65 | 414.48 | 416.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 414.00 | 414.48 | 416.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 411.50 | 414.19 | 415.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:00:00 | 409.80 | 413.07 | 414.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 406.40 | 406.67 | 409.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 407.75 | 404.38 | 404.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 407.75 | 404.38 | 404.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 408.95 | 405.75 | 404.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 408.10 | 411.07 | 408.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 409.40 | 411.07 | 408.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 408.80 | 410.62 | 408.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 411.00 | 410.37 | 408.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 411.85 | 410.38 | 409.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 404.50 | 411.21 | 411.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 404.50 | 411.21 | 411.34 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 418.05 | 411.41 | 410.84 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 407.60 | 411.81 | 412.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 402.15 | 409.22 | 410.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 407.50 | 407.41 | 409.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:30:00 | 407.55 | 407.41 | 409.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 409.10 | 408.00 | 409.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:30:00 | 410.00 | 408.00 | 409.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 408.00 | 408.00 | 409.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 405.80 | 408.00 | 409.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 405.55 | 405.42 | 405.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 405.55 | 405.42 | 405.42 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 404.65 | 405.27 | 405.35 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 407.35 | 405.68 | 405.53 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 404.45 | 405.32 | 405.39 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 409.95 | 406.20 | 405.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 416.45 | 410.10 | 407.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 412.30 | 413.16 | 410.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:00:00 | 412.30 | 413.16 | 410.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 414.50 | 412.92 | 411.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 414.35 | 412.92 | 411.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 408.15 | 412.64 | 411.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 408.15 | 412.64 | 411.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 408.25 | 411.76 | 411.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:15:00 | 408.15 | 411.76 | 411.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 406.50 | 410.24 | 410.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 402.10 | 408.12 | 409.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 406.20 | 405.98 | 408.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 11:45:00 | 401.70 | 405.62 | 407.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 13:30:00 | 401.10 | 405.26 | 407.18 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 407.90 | 405.79 | 407.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 407.90 | 405.79 | 407.24 | SL hit (close>ema400) qty=1.00 sl=407.24 alert=retest1 |

### Cycle 101 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 388.45 | 383.89 | 383.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 12:15:00 | 390.30 | 385.76 | 384.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 392.15 | 394.93 | 392.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 391.00 | 394.93 | 392.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 391.75 | 394.29 | 392.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 391.75 | 394.29 | 392.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 392.30 | 393.89 | 392.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:45:00 | 395.65 | 394.37 | 392.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 395.50 | 395.63 | 394.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 400.15 | 402.16 | 402.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 400.15 | 402.16 | 402.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 398.45 | 401.42 | 401.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 387.80 | 385.40 | 388.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:00:00 | 387.80 | 385.40 | 388.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 390.00 | 386.32 | 388.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 390.00 | 386.32 | 388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 387.95 | 386.65 | 388.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 387.00 | 386.99 | 388.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 14:15:00 | 390.35 | 387.00 | 387.27 | SL hit (close>static) qty=1.00 sl=390.25 alert=retest2 |

### Cycle 103 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 390.50 | 387.70 | 387.57 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 380.60 | 386.28 | 386.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 378.10 | 384.64 | 386.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 378.45 | 378.16 | 381.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:45:00 | 379.15 | 378.16 | 381.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 377.10 | 378.01 | 380.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 376.80 | 378.01 | 380.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 376.20 | 377.84 | 380.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 376.50 | 378.77 | 379.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 373.55 | 377.30 | 378.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.96 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.39 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 357.68 | 366.05 | 370.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 365.00 | 363.82 | 367.05 | SL hit (close>ema200) qty=0.50 sl=363.82 alert=retest2 |

### Cycle 105 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 375.00 | 368.45 | 367.92 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 365.45 | 367.82 | 368.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 362.95 | 366.19 | 367.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 13:15:00 | 343.10 | 341.73 | 347.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 14:45:00 | 339.65 | 341.29 | 346.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 348.00 | 342.56 | 346.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 348.00 | 342.56 | 346.48 | SL hit (close>ema400) qty=1.00 sl=346.48 alert=retest1 |

### Cycle 107 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 354.40 | 346.02 | 345.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 357.20 | 351.35 | 348.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 349.75 | 351.03 | 348.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:30:00 | 350.95 | 351.03 | 348.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 349.35 | 350.70 | 348.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 354.65 | 350.70 | 348.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 352.60 | 361.75 | 362.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 352.60 | 361.75 | 362.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 351.90 | 357.11 | 359.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 360.00 | 357.69 | 359.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 360.00 | 357.69 | 359.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 366.00 | 359.35 | 360.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 402.60 | 359.35 | 360.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 405.00 | 368.48 | 364.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 424.55 | 419.19 | 415.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 427.00 | 427.30 | 422.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 426.00 | 427.30 | 422.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 427.35 | 427.31 | 423.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 432.75 | 427.69 | 423.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 425.30 | 427.59 | 427.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 425.30 | 427.59 | 427.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 10:15:00 | 421.80 | 425.89 | 426.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 15:15:00 | 422.70 | 422.63 | 424.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:15:00 | 418.45 | 422.63 | 424.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 428.15 | 418.11 | 418.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 428.15 | 418.11 | 418.90 | SL hit (close>ema400) qty=1.00 sl=418.90 alert=retest1 |

### Cycle 111 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 424.40 | 420.31 | 419.82 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 417.10 | 419.31 | 419.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 410.25 | 416.97 | 418.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 409.75 | 409.20 | 411.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 405.40 | 408.16 | 410.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 406.05 | 407.02 | 409.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-02 09:15:00 | 364.86 | 401.07 | 404.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 412.00 | 388.02 | 387.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 13:15:00 | 418.40 | 397.07 | 392.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 11:15:00 | 419.45 | 419.76 | 407.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 12:00:00 | 419.45 | 419.76 | 407.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 388.30 | 412.07 | 406.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 388.30 | 412.07 | 406.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 398.80 | 409.42 | 405.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 376.05 | 409.42 | 405.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 375.25 | 402.58 | 403.08 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 390.70 | 386.99 | 386.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 09:15:00 | 399.25 | 390.26 | 388.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 387.60 | 391.55 | 389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 387.60 | 391.55 | 389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 388.80 | 391.00 | 389.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 385.40 | 391.00 | 389.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 384.00 | 389.60 | 388.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 384.00 | 389.60 | 388.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 384.65 | 388.61 | 388.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 378.50 | 388.61 | 388.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 377.75 | 386.44 | 387.58 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 388.40 | 386.50 | 386.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 389.90 | 387.12 | 386.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 387.70 | 392.92 | 391.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 392.85 | 392.03 | 390.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 384.70 | 389.70 | 390.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 384.70 | 389.70 | 390.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 383.10 | 388.38 | 389.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 370.25 | 369.05 | 375.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 368.70 | 368.88 | 374.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 383.30 | 372.27 | 375.16 | SL hit (close>static) qty=1.00 sl=378.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 388.70 | 378.22 | 377.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 389.00 | 380.38 | 378.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 383.05 | 384.09 | 381.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 379.40 | 384.09 | 381.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 377.80 | 382.83 | 381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 377.80 | 382.83 | 381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 376.75 | 381.62 | 380.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 376.65 | 381.62 | 380.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 375.05 | 379.20 | 379.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 373.25 | 377.15 | 378.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 377.65 | 368.82 | 371.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 376.15 | 368.82 | 371.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 379.50 | 370.95 | 372.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 379.80 | 370.95 | 372.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 384.15 | 375.62 | 374.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 384.95 | 379.57 | 376.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 372.15 | 378.09 | 376.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:45:00 | 380.00 | 378.23 | 376.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 380.10 | 380.79 | 378.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 378.25 | 380.79 | 378.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 418.00 | 409.59 | 407.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 418.20 | 420.64 | 420.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 415.45 | 419.60 | 420.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 419.95 | 419.17 | 420.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 419.95 | 419.17 | 420.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 419.20 | 419.17 | 419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 419.60 | 419.17 | 419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 420.35 | 419.41 | 420.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 420.35 | 419.41 | 420.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 420.00 | 419.53 | 420.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 421.10 | 419.53 | 420.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 419.70 | 419.56 | 419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 422.30 | 419.56 | 419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 429.00 | 421.45 | 420.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 436.85 | 429.04 | 427.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 427.50 | 429.99 | 428.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 427.50 | 429.99 | 428.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 425.55 | 429.10 | 428.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 425.55 | 429.10 | 428.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 427.55 | 428.79 | 427.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 432.00 | 428.79 | 427.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 10:30:00 | 428.80 | 428.75 | 428.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 425.85 | 429.38 | 429.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 425.85 | 429.38 | 429.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 422.65 | 425.65 | 427.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 13:15:00 | 425.25 | 424.94 | 426.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 425.25 | 424.94 | 426.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 427.70 | 425.49 | 426.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 429.40 | 425.49 | 426.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 430.00 | 426.40 | 426.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 427.85 | 426.40 | 426.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 427.40 | 426.90 | 427.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:45:00 | 427.95 | 426.90 | 427.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 428.25 | 427.17 | 427.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 427.80 | 427.17 | 427.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 427.75 | 427.29 | 427.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 432.05 | 428.35 | 427.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 429.30 | 429.49 | 428.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 429.00 | 429.49 | 428.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 430.70 | 429.73 | 428.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 430.70 | 429.73 | 428.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 427.00 | 429.47 | 428.91 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 09:30:00 | 205.70 | 2024-05-29 12:15:00 | 209.75 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2024-06-11 12:45:00 | 250.80 | 2024-06-13 09:15:00 | 245.30 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2024-06-12 09:15:00 | 250.86 | 2024-06-13 09:15:00 | 245.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest1 | 2024-06-12 11:30:00 | 248.90 | 2024-06-13 09:15:00 | 245.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest1 | 2024-06-12 12:30:00 | 248.70 | 2024-06-13 09:15:00 | 245.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-06-13 14:15:00 | 251.00 | 2024-06-25 14:15:00 | 264.36 | STOP_HIT | 1.00 | 5.32% |
| SELL | retest2 | 2024-06-27 14:00:00 | 261.00 | 2024-07-01 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-07-01 11:45:00 | 259.85 | 2024-07-01 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-08-09 09:15:00 | 310.05 | 2024-08-13 14:15:00 | 305.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-09 10:45:00 | 308.45 | 2024-08-13 14:15:00 | 305.30 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-08-09 14:00:00 | 310.05 | 2024-08-13 14:15:00 | 305.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-12 09:15:00 | 310.05 | 2024-08-13 14:15:00 | 305.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-13 09:15:00 | 312.30 | 2024-08-13 15:15:00 | 303.50 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-08-13 10:45:00 | 312.35 | 2024-08-13 15:15:00 | 303.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-08-23 09:15:00 | 351.95 | 2024-09-02 09:15:00 | 387.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-13 09:45:00 | 406.90 | 2024-09-16 10:15:00 | 447.04 | TARGET_HIT | 1.00 | 9.86% |
| BUY | retest2 | 2024-09-13 12:00:00 | 406.40 | 2024-09-16 10:15:00 | 446.71 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2024-09-13 13:30:00 | 406.10 | 2024-09-16 10:15:00 | 446.82 | TARGET_HIT | 1.00 | 10.03% |
| BUY | retest2 | 2024-09-13 14:15:00 | 406.20 | 2024-09-17 09:15:00 | 447.59 | TARGET_HIT | 1.00 | 10.19% |
| SELL | retest2 | 2024-09-30 15:00:00 | 398.10 | 2024-10-07 09:15:00 | 378.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 400.90 | 2024-10-07 09:15:00 | 380.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 15:00:00 | 398.10 | 2024-10-07 12:15:00 | 390.10 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2024-10-01 10:15:00 | 400.90 | 2024-10-07 12:15:00 | 390.10 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2024-10-10 15:15:00 | 400.30 | 2024-10-14 09:15:00 | 393.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-10-11 09:30:00 | 400.30 | 2024-10-14 09:15:00 | 393.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-10-11 11:30:00 | 402.75 | 2024-10-14 09:15:00 | 393.90 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-10-18 15:15:00 | 421.10 | 2024-10-21 15:15:00 | 410.50 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-10-21 09:45:00 | 420.95 | 2024-10-21 15:15:00 | 410.50 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-10-24 14:15:00 | 388.95 | 2024-10-24 14:15:00 | 350.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 365.10 | 2024-11-18 09:15:00 | 346.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 365.10 | 2024-11-19 09:15:00 | 353.75 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-12-18 11:00:00 | 418.80 | 2024-12-26 09:15:00 | 397.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 415.15 | 2024-12-26 09:15:00 | 397.91 | PARTIAL | 0.50 | 4.15% |
| SELL | retest2 | 2024-12-19 14:15:00 | 418.85 | 2024-12-26 09:15:00 | 397.24 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-12-18 11:00:00 | 418.80 | 2024-12-27 09:15:00 | 406.00 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2024-12-19 09:15:00 | 415.15 | 2024-12-27 09:15:00 | 406.00 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2024-12-19 14:15:00 | 418.85 | 2024-12-27 09:15:00 | 406.00 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2024-12-20 09:45:00 | 418.15 | 2024-12-30 09:15:00 | 410.55 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-01-01 09:15:00 | 428.50 | 2025-01-06 10:15:00 | 408.05 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest1 | 2025-01-09 09:15:00 | 411.95 | 2025-01-10 09:15:00 | 391.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-09 09:15:00 | 411.95 | 2025-01-13 09:15:00 | 405.50 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2025-01-13 11:00:00 | 394.50 | 2025-01-15 14:15:00 | 400.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-15 13:45:00 | 400.10 | 2025-01-15 14:15:00 | 400.45 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-01-21 10:15:00 | 389.05 | 2025-01-23 10:15:00 | 397.70 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-21 12:00:00 | 390.40 | 2025-01-23 10:15:00 | 397.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-01-30 13:15:00 | 367.10 | 2025-01-31 10:15:00 | 379.10 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-02-07 11:15:00 | 399.40 | 2025-02-11 10:15:00 | 387.95 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-02-10 10:30:00 | 399.40 | 2025-02-11 10:15:00 | 387.95 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-02-10 12:30:00 | 399.00 | 2025-02-11 10:15:00 | 387.95 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-02-10 13:15:00 | 398.70 | 2025-02-11 10:15:00 | 387.95 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-02-21 09:15:00 | 388.40 | 2025-02-21 09:15:00 | 380.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-03-27 14:15:00 | 365.85 | 2025-03-27 14:15:00 | 375.35 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-03 09:15:00 | 364.40 | 2025-04-07 09:15:00 | 327.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 11:00:00 | 357.45 | 2025-04-25 10:15:00 | 351.60 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-04-17 14:15:00 | 357.80 | 2025-04-25 10:15:00 | 351.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-04-21 12:45:00 | 357.10 | 2025-04-25 10:15:00 | 351.60 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-04-21 14:45:00 | 357.25 | 2025-04-25 10:15:00 | 351.60 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-06 12:30:00 | 342.40 | 2025-05-07 14:15:00 | 350.75 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-05-07 11:45:00 | 342.85 | 2025-05-07 14:15:00 | 350.75 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-05-09 10:45:00 | 351.00 | 2025-05-19 12:15:00 | 386.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 12:00:00 | 351.50 | 2025-05-19 12:15:00 | 386.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 449.60 | 2025-06-19 11:15:00 | 427.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:30:00 | 449.60 | 2025-06-20 09:15:00 | 436.30 | STOP_HIT | 0.50 | 2.96% |
| BUY | retest2 | 2025-07-04 14:45:00 | 490.50 | 2025-07-07 09:15:00 | 480.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-18 11:30:00 | 510.65 | 2025-07-21 09:15:00 | 499.10 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-07-25 10:15:00 | 486.35 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-07-25 12:45:00 | 488.85 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-07-25 13:45:00 | 486.70 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-28 09:15:00 | 487.10 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-07-28 12:15:00 | 483.35 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-07-28 14:45:00 | 484.00 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-07-29 10:15:00 | 482.30 | 2025-07-29 12:15:00 | 496.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-08-06 10:30:00 | 464.35 | 2025-08-11 09:15:00 | 441.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 09:15:00 | 464.40 | 2025-08-11 09:15:00 | 441.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 10:30:00 | 464.35 | 2025-08-11 10:15:00 | 455.95 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-08-07 09:15:00 | 464.40 | 2025-08-11 10:15:00 | 455.95 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2025-08-11 15:15:00 | 464.90 | 2025-08-12 09:15:00 | 463.20 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-08-28 09:15:00 | 423.10 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-08-29 10:00:00 | 423.15 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-08-29 12:45:00 | 424.60 | 2025-09-01 15:15:00 | 431.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-05 15:15:00 | 444.70 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-08 09:30:00 | 447.40 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-09-08 11:00:00 | 445.00 | 2025-09-08 11:15:00 | 440.85 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-08 14:15:00 | 444.95 | 2025-09-09 09:15:00 | 440.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-17 11:00:00 | 465.70 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-18 10:00:00 | 464.95 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-18 15:00:00 | 465.00 | 2025-09-22 12:15:00 | 459.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-03 10:45:00 | 401.35 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-10-06 09:15:00 | 399.30 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-07 09:30:00 | 400.85 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-10-07 10:00:00 | 399.60 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-08 09:30:00 | 393.55 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-08 10:00:00 | 393.25 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-09 09:30:00 | 392.30 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-10-09 10:00:00 | 393.00 | 2025-10-10 11:15:00 | 400.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-16 13:15:00 | 413.15 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-10-16 14:00:00 | 413.15 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-10-17 13:45:00 | 412.20 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-10-20 10:45:00 | 416.80 | 2025-10-27 15:15:00 | 421.90 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-10-30 12:30:00 | 414.80 | 2025-10-30 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-31 10:15:00 | 414.50 | 2025-10-31 13:15:00 | 419.65 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-04 12:00:00 | 409.80 | 2025-11-11 15:15:00 | 407.75 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-11-06 15:00:00 | 406.40 | 2025-11-11 15:15:00 | 407.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-13 12:30:00 | 411.00 | 2025-11-18 09:15:00 | 404.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-11-14 09:15:00 | 411.85 | 2025-11-18 09:15:00 | 404.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-24 09:15:00 | 405.80 | 2025-11-27 11:15:00 | 405.55 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest1 | 2025-12-04 11:45:00 | 401.70 | 2025-12-04 14:15:00 | 407.90 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-12-04 13:30:00 | 401.10 | 2025-12-04 14:15:00 | 407.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-05 09:15:00 | 403.65 | 2025-12-09 09:15:00 | 363.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 12:45:00 | 395.65 | 2025-12-29 09:15:00 | 400.15 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-12-19 11:30:00 | 395.50 | 2025-12-29 09:15:00 | 400.15 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2026-01-01 10:45:00 | 387.00 | 2026-01-02 14:15:00 | 390.35 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-07 09:15:00 | 376.80 | 2026-01-12 09:15:00 | 357.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 376.20 | 2026-01-12 09:15:00 | 357.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 376.50 | 2026-01-12 09:15:00 | 357.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 376.80 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-01-07 10:15:00 | 376.20 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-01-08 11:00:00 | 376.50 | 2026-01-12 15:15:00 | 365.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-01-08 13:00:00 | 373.55 | 2026-01-13 09:15:00 | 354.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:00:00 | 373.55 | 2026-01-13 09:15:00 | 364.85 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2026-01-21 14:45:00 | 339.65 | 2026-01-22 09:15:00 | 348.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-01-22 11:30:00 | 344.25 | 2026-01-22 12:15:00 | 348.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-23 09:30:00 | 344.20 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-23 10:15:00 | 344.15 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-01-23 15:00:00 | 343.75 | 2026-01-27 11:15:00 | 354.40 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-28 11:15:00 | 354.65 | 2026-02-02 10:15:00 | 352.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-02-13 11:15:00 | 432.75 | 2026-02-17 14:15:00 | 425.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest1 | 2026-02-19 09:15:00 | 418.45 | 2026-02-23 09:15:00 | 428.15 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-02-26 11:30:00 | 405.40 | 2026-03-02 09:15:00 | 364.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 406.05 | 2026-03-02 09:15:00 | 365.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 11:15:00 | 392.85 | 2026-03-19 13:15:00 | 384.70 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-24 10:30:00 | 368.70 | 2026-03-24 12:15:00 | 383.30 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2026-04-02 12:45:00 | 380.00 | 2026-04-15 09:15:00 | 418.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:30:00 | 380.10 | 2026-04-15 09:15:00 | 418.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:15:00 | 378.25 | 2026-04-15 09:15:00 | 416.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 09:15:00 | 432.00 | 2026-05-05 10:15:00 | 425.85 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-30 10:30:00 | 428.80 | 2026-05-05 10:15:00 | 425.85 | STOP_HIT | 1.00 | -0.69% |
