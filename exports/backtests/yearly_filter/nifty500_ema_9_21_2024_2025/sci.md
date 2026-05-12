# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 339.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 123 |
| ALERT1 | 89 |
| ALERT2 | 89 |
| ALERT2_SKIP | 42 |
| ALERT3 | 255 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 109 |
| PARTIAL | 30 |
| TARGET_HIT | 13 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 78 / 67
- **Target hits / Stop hits / Partials:** 13 / 102 / 30
- **Avg / median % per leg:** 1.89% / 0.53%
- **Sum % (uncompounded):** 274.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 19 | 35.2% | 5 | 48 | 1 | 0.41% | 22.1% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 7 | 1 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 46 | 16 | 34.8% | 5 | 41 | 0 | 0.48% | 22.2% |
| SELL (all) | 91 | 59 | 64.8% | 8 | 54 | 29 | 2.77% | 252.0% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.51% | 18.1% |
| SELL @ 3rd Alert (retest2) | 87 | 55 | 63.2% | 8 | 52 | 27 | 2.69% | 233.9% |
| retest1 (combined) | 12 | 7 | 58.3% | 0 | 9 | 3 | 1.50% | 18.0% |
| retest2 (combined) | 133 | 71 | 53.4% | 13 | 93 | 27 | 1.93% | 256.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 209.35 | 206.92 | 206.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 219.50 | 210.27 | 208.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 230.75 | 232.52 | 228.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 12:00:00 | 230.75 | 232.52 | 228.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 242.30 | 234.23 | 230.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:45:00 | 243.40 | 235.86 | 231.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 247.25 | 239.19 | 236.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 09:15:00 | 267.74 | 243.97 | 239.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 251.30 | 256.40 | 256.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 248.10 | 254.74 | 255.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 251.95 | 251.25 | 253.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 251.95 | 251.25 | 253.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 251.95 | 251.25 | 253.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 253.25 | 251.25 | 253.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 250.90 | 250.86 | 252.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:45:00 | 251.50 | 250.86 | 252.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 251.20 | 251.01 | 252.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 251.20 | 251.01 | 252.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 250.30 | 243.36 | 245.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 250.30 | 243.36 | 245.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 245.65 | 243.82 | 245.74 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 261.10 | 247.54 | 247.11 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 220.70 | 247.82 | 249.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 214.10 | 241.08 | 246.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 222.00 | 221.99 | 231.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 222.00 | 221.99 | 231.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 241.05 | 226.62 | 230.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 241.05 | 226.62 | 230.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 241.25 | 229.55 | 231.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 241.65 | 229.55 | 231.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 242.00 | 233.76 | 233.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 12:15:00 | 247.86 | 245.54 | 243.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 256.40 | 257.21 | 252.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 256.10 | 257.21 | 252.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 254.30 | 256.09 | 252.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:45:00 | 255.52 | 255.95 | 253.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 281.07 | 269.34 | 263.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 267.35 | 272.45 | 272.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 266.11 | 271.19 | 272.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 10:15:00 | 266.00 | 263.76 | 265.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 10:15:00 | 266.00 | 263.76 | 265.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 266.00 | 263.76 | 265.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 266.00 | 263.76 | 265.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 262.85 | 263.58 | 265.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:00:00 | 262.55 | 263.38 | 265.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 261.40 | 263.08 | 264.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 14:15:00 | 249.42 | 255.94 | 258.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 258.31 | 255.94 | 258.19 | SL hit (close>ema200) qty=0.50 sl=255.94 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 262.00 | 256.00 | 255.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 264.60 | 257.72 | 256.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 12:15:00 | 269.70 | 269.86 | 266.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:00:00 | 269.70 | 269.86 | 266.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 269.10 | 270.23 | 267.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 269.10 | 270.23 | 267.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 270.75 | 271.33 | 269.69 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 262.10 | 268.67 | 269.08 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 274.25 | 268.63 | 268.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 319.70 | 280.22 | 273.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 329.95 | 339.13 | 323.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:45:00 | 329.10 | 339.13 | 323.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 328.10 | 331.74 | 327.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 328.10 | 331.74 | 327.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 327.30 | 330.86 | 327.66 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 314.45 | 325.05 | 325.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 306.80 | 313.97 | 318.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 311.05 | 310.80 | 315.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 13:15:00 | 311.05 | 310.80 | 315.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 311.05 | 310.80 | 315.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 317.20 | 310.80 | 315.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 328.30 | 312.74 | 315.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 334.85 | 312.74 | 315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 317.75 | 313.74 | 315.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 328.20 | 313.74 | 315.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 318.30 | 316.21 | 316.14 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 312.35 | 315.77 | 316.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 298.60 | 312.34 | 314.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 289.80 | 283.61 | 287.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 289.80 | 283.61 | 287.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 289.80 | 283.61 | 287.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 290.90 | 283.61 | 287.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 284.90 | 283.87 | 287.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 11:45:00 | 283.30 | 283.10 | 286.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 10:00:00 | 281.35 | 281.96 | 284.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 286.55 | 285.72 | 285.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 286.55 | 285.72 | 285.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 292.70 | 287.54 | 286.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 13:15:00 | 288.55 | 288.64 | 287.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 13:15:00 | 288.55 | 288.64 | 287.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 288.55 | 288.64 | 287.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:30:00 | 287.85 | 288.64 | 287.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 286.75 | 288.26 | 287.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 287.20 | 288.26 | 287.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 287.00 | 288.01 | 287.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 283.55 | 288.01 | 287.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 284.70 | 286.65 | 286.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 282.85 | 284.90 | 285.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 15:15:00 | 285.00 | 284.92 | 285.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 09:15:00 | 266.45 | 284.92 | 285.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 14:15:00 | 253.13 | 263.41 | 270.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 262.85 | 259.72 | 264.77 | SL hit (close>ema200) qty=0.50 sl=259.72 alert=retest1 |

### Cycle 15 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 280.00 | 262.96 | 262.29 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 262.35 | 266.80 | 267.40 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 270.50 | 266.49 | 266.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 271.45 | 267.48 | 266.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 272.90 | 273.40 | 271.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 272.90 | 273.40 | 271.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 269.00 | 272.52 | 271.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 269.30 | 272.52 | 271.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 269.25 | 271.87 | 271.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 268.60 | 271.87 | 271.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 269.55 | 270.94 | 270.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 269.55 | 270.94 | 270.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 271.50 | 271.05 | 270.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 276.25 | 271.05 | 270.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 275.10 | 272.94 | 272.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:15:00 | 275.55 | 274.57 | 273.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:00:00 | 274.85 | 275.16 | 274.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 274.80 | 275.09 | 274.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 274.80 | 275.09 | 274.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 274.10 | 274.89 | 274.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 274.10 | 274.89 | 274.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 274.60 | 274.83 | 274.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 274.30 | 274.83 | 274.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 274.70 | 274.81 | 274.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 274.70 | 274.81 | 274.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 274.00 | 274.65 | 274.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 275.75 | 274.65 | 274.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 12:15:00 | 272.95 | 274.27 | 274.22 | SL hit (close<static) qty=1.00 sl=273.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 272.70 | 273.95 | 274.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 271.75 | 273.07 | 273.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 267.50 | 266.99 | 269.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 267.50 | 266.99 | 269.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 267.50 | 266.99 | 269.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 267.50 | 266.99 | 269.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 269.40 | 267.06 | 268.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 269.40 | 267.06 | 268.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 269.10 | 267.47 | 268.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 267.65 | 267.47 | 268.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 265.35 | 267.04 | 268.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 264.50 | 267.04 | 268.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 270.85 | 266.58 | 267.03 | SL hit (close>static) qty=1.00 sl=270.50 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 273.65 | 267.99 | 267.63 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 267.30 | 269.28 | 269.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 267.10 | 268.84 | 269.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 255.00 | 252.38 | 256.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 255.00 | 252.38 | 256.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 255.00 | 252.38 | 256.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 256.30 | 252.38 | 256.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 254.00 | 253.86 | 255.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:45:00 | 252.65 | 253.67 | 255.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 12:15:00 | 240.02 | 246.14 | 249.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 245.45 | 245.26 | 248.42 | SL hit (close>ema200) qty=0.50 sl=245.26 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 250.45 | 245.13 | 244.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 259.80 | 250.19 | 247.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 12:15:00 | 266.05 | 266.19 | 260.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:45:00 | 265.15 | 266.19 | 260.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 264.10 | 266.35 | 263.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 264.20 | 266.35 | 263.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 263.30 | 266.04 | 264.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 263.05 | 266.04 | 264.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 266.45 | 266.12 | 264.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 269.80 | 266.49 | 264.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:00:00 | 266.90 | 266.57 | 265.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 270.20 | 267.29 | 265.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 267.00 | 267.55 | 266.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 266.75 | 267.39 | 266.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 266.50 | 267.39 | 266.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 265.05 | 266.92 | 266.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:45:00 | 264.85 | 266.92 | 266.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 266.00 | 266.74 | 266.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:15:00 | 266.70 | 266.74 | 266.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 262.60 | 266.32 | 266.15 | SL hit (close<static) qty=1.00 sl=262.80 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 262.50 | 265.56 | 265.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 15:15:00 | 260.50 | 262.84 | 264.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 15:15:00 | 260.90 | 260.52 | 262.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:15:00 | 255.10 | 260.52 | 262.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 242.34 | 246.10 | 250.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 238.00 | 237.94 | 243.03 | SL hit (close>ema200) qty=0.50 sl=237.94 alert=retest1 |

### Cycle 23 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 247.65 | 243.45 | 242.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 241.50 | 243.88 | 244.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 239.70 | 243.04 | 243.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 242.70 | 240.38 | 241.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 242.70 | 240.38 | 241.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 242.70 | 240.38 | 241.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 242.70 | 240.38 | 241.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 241.70 | 240.65 | 241.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 241.15 | 240.65 | 241.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:45:00 | 240.35 | 240.49 | 241.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 245.80 | 241.75 | 241.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 245.80 | 241.75 | 241.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 240.60 | 242.80 | 242.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 239.85 | 242.21 | 242.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 10:15:00 | 218.20 | 216.76 | 221.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:30:00 | 220.80 | 216.76 | 221.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 206.60 | 204.96 | 206.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 206.60 | 204.96 | 206.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 207.25 | 205.41 | 206.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 210.85 | 205.41 | 206.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 211.45 | 206.62 | 207.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 211.45 | 206.62 | 207.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 215.85 | 208.47 | 208.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 217.80 | 213.36 | 210.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 214.55 | 214.67 | 212.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 214.55 | 214.67 | 212.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 212.06 | 216.04 | 214.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 212.06 | 216.04 | 214.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 212.20 | 215.28 | 214.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 211.73 | 215.28 | 214.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 214.58 | 214.98 | 214.20 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 212.59 | 213.80 | 213.86 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 216.23 | 214.05 | 213.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 218.11 | 215.35 | 214.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 221.50 | 221.67 | 219.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 221.50 | 221.67 | 219.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 215.99 | 220.32 | 219.46 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 212.51 | 217.73 | 218.37 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 238.89 | 219.74 | 218.71 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 215.24 | 223.54 | 224.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 211.20 | 217.26 | 220.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 15:15:00 | 211.60 | 209.23 | 211.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:15:00 | 214.81 | 209.23 | 211.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 215.64 | 210.51 | 212.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 214.64 | 210.51 | 212.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 211.26 | 208.52 | 209.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 211.26 | 208.52 | 209.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 212.09 | 209.23 | 210.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 212.77 | 209.23 | 210.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 217.40 | 210.87 | 210.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 15:15:00 | 221.00 | 215.67 | 213.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 232.34 | 233.70 | 230.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 232.34 | 233.70 | 230.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 232.20 | 233.11 | 230.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 231.42 | 233.11 | 230.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 233.55 | 233.95 | 232.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 239.39 | 233.94 | 232.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:00:00 | 236.56 | 235.00 | 233.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:45:00 | 236.75 | 235.42 | 233.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:45:00 | 236.64 | 235.89 | 234.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 236.99 | 237.46 | 236.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 235.58 | 237.46 | 236.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 236.50 | 237.27 | 236.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 235.58 | 237.27 | 236.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 236.92 | 237.20 | 236.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:30:00 | 238.39 | 237.16 | 236.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 238.17 | 236.85 | 236.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 238.94 | 236.90 | 236.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:30:00 | 238.15 | 238.83 | 238.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 237.75 | 238.61 | 238.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 236.59 | 238.61 | 238.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 236.16 | 238.12 | 238.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 236.16 | 238.12 | 238.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 237.64 | 238.02 | 238.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 237.64 | 238.02 | 238.04 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 240.00 | 238.23 | 238.05 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 237.28 | 238.11 | 238.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 233.54 | 236.82 | 237.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 230.90 | 230.68 | 233.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 230.90 | 230.68 | 233.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 233.40 | 231.19 | 232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 234.44 | 231.19 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 232.67 | 231.49 | 232.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 233.50 | 231.49 | 232.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 232.50 | 231.69 | 232.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 232.80 | 231.69 | 232.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 233.16 | 231.98 | 232.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 233.34 | 231.98 | 232.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 232.68 | 232.12 | 232.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 232.45 | 232.12 | 232.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 232.58 | 232.21 | 232.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 231.59 | 232.01 | 232.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 220.83 | 225.61 | 228.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 220.95 | 225.61 | 228.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 220.01 | 222.86 | 226.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 220.42 | 220.21 | 222.70 | SL hit (close>ema200) qty=0.50 sl=220.21 alert=retest2 |

### Cycle 37 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 211.35 | 208.34 | 207.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 212.00 | 210.36 | 209.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 214.26 | 214.83 | 213.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 210.70 | 214.01 | 213.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 210.70 | 214.01 | 213.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 211.39 | 214.01 | 213.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 206.42 | 212.49 | 212.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 204.57 | 209.48 | 211.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 207.70 | 207.58 | 209.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 207.70 | 207.58 | 209.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 208.80 | 208.14 | 209.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 205.58 | 208.14 | 209.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 210.65 | 208.03 | 208.34 | SL hit (close>static) qty=1.00 sl=210.19 alert=retest2 |

### Cycle 39 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 198.80 | 195.27 | 195.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 201.40 | 198.22 | 196.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 205.53 | 206.23 | 203.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 205.53 | 206.23 | 203.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 203.09 | 205.60 | 203.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 203.09 | 205.60 | 203.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 203.85 | 205.25 | 203.86 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 196.21 | 202.13 | 202.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 193.60 | 199.25 | 201.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 199.31 | 197.87 | 199.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 199.31 | 197.87 | 199.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 199.31 | 197.87 | 199.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 199.31 | 197.87 | 199.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 200.29 | 198.35 | 199.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 200.48 | 198.35 | 199.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 199.64 | 198.61 | 199.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 199.09 | 198.61 | 199.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 199.41 | 198.77 | 199.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 198.60 | 198.89 | 199.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 188.67 | 193.85 | 196.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 178.74 | 186.32 | 190.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 191.24 | 189.00 | 188.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 194.06 | 190.01 | 189.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 190.11 | 191.16 | 190.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 190.11 | 191.16 | 190.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 190.11 | 191.16 | 190.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 189.85 | 191.16 | 190.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 190.93 | 191.11 | 190.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 190.99 | 191.11 | 190.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 195.63 | 199.98 | 197.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 194.53 | 199.98 | 197.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 196.55 | 199.29 | 197.39 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 195.04 | 196.43 | 196.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 191.50 | 194.83 | 195.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 195.49 | 193.45 | 194.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 195.49 | 193.45 | 194.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 195.49 | 193.45 | 194.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 196.74 | 193.45 | 194.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 195.20 | 193.80 | 194.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 195.88 | 193.80 | 194.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 196.00 | 194.88 | 194.82 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 194.25 | 194.75 | 194.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 193.64 | 194.54 | 194.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 194.78 | 194.59 | 194.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 194.78 | 194.59 | 194.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 194.78 | 194.59 | 194.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 194.78 | 194.59 | 194.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 195.00 | 194.67 | 194.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 192.21 | 194.67 | 194.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 192.05 | 194.15 | 194.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 191.21 | 194.15 | 194.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:45:00 | 190.81 | 192.45 | 193.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 181.55 | 192.16 | 193.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 181.65 | 189.93 | 192.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 181.27 | 189.93 | 192.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 172.09 | 180.23 | 185.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 158.48 | 155.11 | 155.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 159.34 | 155.96 | 155.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 160.32 | 160.93 | 159.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 160.32 | 160.93 | 159.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 160.00 | 160.82 | 159.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 160.00 | 160.82 | 159.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 158.08 | 160.44 | 159.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 160.57 | 159.86 | 159.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 12:45:00 | 159.78 | 159.84 | 159.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 158.93 | 159.48 | 159.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 158.93 | 159.48 | 159.49 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 160.35 | 159.62 | 159.55 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 157.86 | 159.21 | 159.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 157.08 | 158.79 | 159.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 144.37 | 143.83 | 147.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:45:00 | 143.22 | 143.83 | 147.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 147.16 | 144.58 | 146.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 147.16 | 144.58 | 146.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 147.00 | 145.06 | 146.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:45:00 | 145.83 | 145.08 | 146.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 150.22 | 146.74 | 146.92 | SL hit (close>static) qty=1.00 sl=148.03 alert=retest2 |

### Cycle 49 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 150.85 | 147.56 | 147.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 151.97 | 149.70 | 148.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 156.14 | 157.11 | 155.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 156.14 | 157.11 | 155.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 156.14 | 157.11 | 155.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 155.24 | 157.11 | 155.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 154.68 | 156.63 | 155.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 154.68 | 156.63 | 155.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 154.41 | 156.18 | 155.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 154.41 | 156.18 | 155.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 154.22 | 155.79 | 155.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 154.25 | 155.79 | 155.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 151.20 | 153.96 | 154.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 149.10 | 150.70 | 151.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 148.80 | 147.22 | 148.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 148.80 | 147.22 | 148.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 148.80 | 147.22 | 148.70 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 156.31 | 150.15 | 149.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 11:15:00 | 157.33 | 151.58 | 150.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 171.76 | 172.15 | 164.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:30:00 | 176.72 | 172.23 | 168.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 11:45:00 | 175.00 | 175.80 | 173.07 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 12:30:00 | 174.99 | 175.61 | 173.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 13:30:00 | 174.98 | 175.49 | 173.39 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 175.00 | 175.19 | 173.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 172.97 | 175.19 | 173.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 172.02 | 174.56 | 173.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 172.02 | 174.56 | 173.60 | SL hit (close<ema400) qty=1.00 sl=173.60 alert=retest1 |

### Cycle 52 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 169.86 | 172.73 | 172.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 167.30 | 171.65 | 172.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 172.95 | 168.90 | 169.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 172.95 | 168.90 | 169.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 172.95 | 168.90 | 169.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 172.95 | 168.90 | 169.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 170.85 | 169.29 | 169.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 170.85 | 169.29 | 169.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 168.39 | 167.44 | 168.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 170.82 | 167.44 | 168.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 167.41 | 167.43 | 168.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 166.44 | 167.44 | 168.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 169.70 | 168.05 | 168.45 | SL hit (close>static) qty=1.00 sl=168.60 alert=retest2 |

### Cycle 53 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 169.60 | 168.82 | 168.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 169.83 | 169.17 | 168.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 165.59 | 170.23 | 170.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 165.59 | 170.23 | 170.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 165.59 | 170.23 | 170.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 165.59 | 170.23 | 170.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 166.80 | 169.55 | 169.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 149.05 | 163.39 | 166.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 160.00 | 157.02 | 161.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 161.39 | 157.02 | 161.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 159.69 | 157.56 | 160.93 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 170.08 | 162.99 | 162.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 172.15 | 166.98 | 165.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 175.00 | 175.19 | 173.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:15:00 | 176.74 | 175.19 | 173.29 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:15:00 | 176.03 | 175.16 | 173.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 177.11 | 175.97 | 174.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 176.70 | 177.41 | 176.71 | SL hit (close<ema400) qty=1.00 sl=176.71 alert=retest1 |

### Cycle 56 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 174.83 | 176.25 | 176.30 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 177.40 | 176.27 | 176.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 184.40 | 177.90 | 176.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 15:15:00 | 180.64 | 180.67 | 179.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-25 09:15:00 | 177.55 | 180.67 | 179.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 175.08 | 179.55 | 178.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 175.08 | 179.55 | 178.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 173.68 | 178.38 | 178.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 174.25 | 178.38 | 178.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 173.70 | 177.44 | 177.85 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 184.17 | 177.95 | 177.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 12:15:00 | 186.92 | 179.74 | 178.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 180.72 | 181.43 | 179.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 179.33 | 180.85 | 179.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 179.33 | 180.85 | 179.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 179.33 | 180.85 | 179.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 178.25 | 180.33 | 179.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 178.25 | 180.33 | 179.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 176.76 | 179.62 | 179.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 177.75 | 179.62 | 179.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 178.02 | 179.05 | 179.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 176.05 | 178.45 | 178.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 11:15:00 | 176.59 | 175.79 | 176.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:45:00 | 176.32 | 175.79 | 176.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 176.27 | 175.89 | 176.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 175.42 | 176.18 | 176.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 15:15:00 | 166.65 | 170.27 | 172.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 170.90 | 170.40 | 171.91 | SL hit (close>ema200) qty=0.50 sl=170.40 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 171.17 | 167.17 | 166.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 173.10 | 168.35 | 167.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 175.75 | 175.79 | 174.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 175.75 | 175.79 | 174.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 172.52 | 174.85 | 174.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 172.52 | 174.85 | 174.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 173.05 | 174.49 | 174.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 177.08 | 174.49 | 174.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 09:15:00 | 194.79 | 186.20 | 181.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 180.00 | 183.25 | 183.39 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 192.40 | 183.92 | 183.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 198.51 | 186.83 | 184.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 200.13 | 200.18 | 196.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:15:00 | 200.24 | 200.18 | 196.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 203.76 | 204.67 | 203.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 203.54 | 204.67 | 203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 204.03 | 204.47 | 203.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 203.42 | 204.47 | 203.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 203.75 | 204.33 | 203.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 203.75 | 204.33 | 203.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 204.53 | 204.37 | 203.45 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 202.11 | 202.93 | 203.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 201.10 | 202.17 | 202.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 203.40 | 202.42 | 202.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 203.40 | 202.42 | 202.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 203.40 | 202.42 | 202.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 205.00 | 202.42 | 202.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 203.07 | 202.55 | 202.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 203.25 | 202.55 | 202.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 203.31 | 202.70 | 202.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 203.62 | 202.70 | 202.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 202.97 | 202.75 | 202.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:15:00 | 205.34 | 202.75 | 202.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 203.61 | 202.93 | 202.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 208.17 | 204.25 | 203.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 212.50 | 212.82 | 210.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 212.36 | 212.82 | 210.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 212.19 | 213.06 | 211.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:45:00 | 215.25 | 213.04 | 211.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 215.80 | 213.51 | 212.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 14:00:00 | 215.37 | 214.21 | 213.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 216.00 | 214.32 | 213.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 213.70 | 214.49 | 213.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 213.70 | 214.49 | 213.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 214.00 | 214.40 | 213.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:15:00 | 213.96 | 214.40 | 213.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 213.80 | 214.28 | 213.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:15:00 | 212.00 | 214.28 | 213.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 209.05 | 213.23 | 213.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 209.05 | 213.23 | 213.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 205.24 | 210.13 | 211.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 216.35 | 210.23 | 211.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 216.35 | 210.23 | 211.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 216.35 | 210.23 | 211.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 216.35 | 210.23 | 211.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 227.66 | 213.71 | 212.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 11:15:00 | 235.40 | 218.05 | 214.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 222.51 | 229.89 | 225.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 222.51 | 229.89 | 225.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 222.51 | 229.89 | 225.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 221.97 | 229.89 | 225.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 221.72 | 228.26 | 225.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:45:00 | 222.00 | 228.26 | 225.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 219.04 | 223.93 | 223.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 216.75 | 222.50 | 223.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 223.77 | 221.63 | 222.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 223.77 | 221.63 | 222.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 223.77 | 221.63 | 222.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 225.35 | 221.63 | 222.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 222.44 | 221.79 | 222.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 220.30 | 221.83 | 222.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 220.16 | 221.60 | 222.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 220.50 | 221.10 | 221.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 219.45 | 221.10 | 221.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 218.40 | 216.94 | 218.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 218.40 | 216.94 | 218.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 219.30 | 217.41 | 218.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 219.32 | 217.41 | 218.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 219.11 | 217.75 | 218.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:15:00 | 219.51 | 217.75 | 218.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 227.64 | 220.19 | 219.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 227.64 | 220.19 | 219.60 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 222.08 | 222.70 | 222.72 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 226.51 | 223.27 | 222.94 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 224.39 | 224.65 | 224.66 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 224.94 | 224.71 | 224.69 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 224.10 | 224.59 | 224.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 223.92 | 224.46 | 224.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 223.22 | 222.74 | 223.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 223.22 | 222.74 | 223.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 223.22 | 222.74 | 223.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 223.38 | 222.74 | 223.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 229.60 | 223.14 | 223.07 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 224.55 | 225.30 | 225.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 223.73 | 224.99 | 225.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 15:15:00 | 223.80 | 223.70 | 224.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 223.65 | 223.70 | 224.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 221.80 | 223.32 | 224.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 221.00 | 223.06 | 223.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 221.40 | 223.06 | 223.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:45:00 | 221.50 | 222.76 | 223.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 221.30 | 222.35 | 223.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 220.00 | 218.71 | 220.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 222.98 | 220.17 | 220.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 222.98 | 220.17 | 220.12 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 219.30 | 220.65 | 220.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 219.00 | 220.32 | 220.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 218.95 | 218.84 | 219.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 218.95 | 218.84 | 219.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 218.95 | 218.84 | 219.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 217.75 | 218.62 | 219.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 222.79 | 219.03 | 218.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 222.79 | 219.03 | 218.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 224.58 | 221.50 | 220.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 222.65 | 222.80 | 221.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:30:00 | 222.52 | 222.80 | 221.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 220.65 | 222.31 | 221.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 220.00 | 222.31 | 221.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 219.45 | 221.74 | 221.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 219.75 | 221.74 | 221.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 218.97 | 221.18 | 221.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 217.83 | 220.14 | 220.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 214.21 | 214.10 | 216.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 214.21 | 214.10 | 216.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 215.77 | 214.86 | 216.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 215.00 | 215.35 | 215.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 216.35 | 215.94 | 215.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 216.35 | 215.94 | 215.94 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 214.15 | 215.58 | 215.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 213.50 | 214.92 | 215.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 211.84 | 211.47 | 212.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 211.84 | 211.47 | 212.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 212.25 | 211.58 | 212.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 215.40 | 211.58 | 212.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 214.18 | 212.10 | 212.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 211.11 | 211.90 | 212.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 200.55 | 203.74 | 205.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 204.20 | 202.45 | 204.06 | SL hit (close>ema200) qty=0.50 sl=202.45 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 208.70 | 204.85 | 204.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 211.88 | 208.33 | 207.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 216.28 | 216.73 | 215.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:30:00 | 216.40 | 216.73 | 215.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 215.45 | 216.41 | 215.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 215.45 | 216.41 | 215.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 215.29 | 216.18 | 215.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 215.16 | 216.18 | 215.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 215.00 | 215.95 | 215.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 215.00 | 215.95 | 215.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 214.41 | 215.29 | 215.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:15:00 | 214.15 | 215.29 | 215.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 214.15 | 215.07 | 215.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 212.16 | 214.48 | 214.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 212.99 | 212.28 | 213.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 212.99 | 212.28 | 213.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 212.99 | 212.28 | 213.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 212.99 | 212.28 | 213.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 215.27 | 212.88 | 213.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 215.27 | 212.88 | 213.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 214.51 | 213.21 | 213.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 214.22 | 213.24 | 213.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 214.68 | 213.67 | 213.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 214.68 | 213.67 | 213.66 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 212.50 | 213.46 | 213.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 15:15:00 | 212.00 | 213.17 | 213.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 214.80 | 213.49 | 213.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 214.80 | 213.49 | 213.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 214.80 | 213.49 | 213.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 215.62 | 213.49 | 213.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 214.72 | 213.74 | 213.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 215.27 | 214.05 | 213.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 214.64 | 219.82 | 218.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 214.64 | 219.82 | 218.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 214.64 | 219.82 | 218.97 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 214.46 | 217.93 | 218.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 210.53 | 215.09 | 216.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 208.84 | 207.35 | 208.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 208.84 | 207.35 | 208.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 208.84 | 207.35 | 208.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 209.25 | 207.35 | 208.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 208.67 | 207.61 | 208.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 208.54 | 207.61 | 208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 208.30 | 207.85 | 208.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 208.86 | 207.85 | 208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 214.31 | 209.21 | 209.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 220.41 | 217.51 | 215.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 218.65 | 219.03 | 217.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:45:00 | 218.76 | 219.03 | 217.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 217.64 | 218.87 | 217.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 217.64 | 218.87 | 217.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 218.20 | 218.74 | 217.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 219.10 | 218.81 | 218.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 218.84 | 218.98 | 218.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:30:00 | 218.54 | 218.73 | 218.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 218.78 | 218.90 | 218.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 223.80 | 225.91 | 223.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 223.13 | 225.91 | 223.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 221.21 | 224.97 | 223.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 221.21 | 224.97 | 223.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 221.59 | 224.30 | 223.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 221.27 | 224.30 | 223.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 222.98 | 223.79 | 223.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 223.68 | 223.79 | 223.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 223.66 | 223.61 | 223.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 224.84 | 228.07 | 228.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 224.84 | 228.07 | 228.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 219.80 | 225.08 | 226.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 222.42 | 221.42 | 223.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 222.42 | 221.42 | 223.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 222.42 | 221.42 | 223.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 222.42 | 221.42 | 223.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 222.50 | 221.63 | 223.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 223.44 | 221.97 | 223.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 223.27 | 222.23 | 223.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 224.50 | 222.23 | 223.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 223.24 | 222.43 | 223.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 223.11 | 222.43 | 223.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 223.68 | 222.68 | 223.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 223.68 | 222.68 | 223.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 224.10 | 222.97 | 223.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 224.10 | 222.97 | 223.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 223.65 | 223.10 | 223.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 224.17 | 223.10 | 223.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 225.29 | 223.59 | 223.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 227.75 | 224.81 | 224.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 226.26 | 226.70 | 225.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 226.26 | 226.70 | 225.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 226.26 | 226.70 | 225.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 225.91 | 226.70 | 225.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 223.95 | 226.15 | 225.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 223.95 | 226.15 | 225.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 223.60 | 225.64 | 225.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 223.92 | 225.64 | 225.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 224.21 | 225.00 | 224.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 224.21 | 225.00 | 224.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 225.00 | 225.00 | 224.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 224.37 | 225.00 | 224.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 222.89 | 224.58 | 224.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 222.34 | 224.13 | 224.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 220.24 | 219.37 | 220.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 220.24 | 219.37 | 220.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 220.24 | 219.37 | 220.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 220.24 | 219.37 | 220.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 221.32 | 219.90 | 220.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 221.32 | 219.90 | 220.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 221.16 | 220.15 | 220.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 223.40 | 220.15 | 220.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 222.05 | 221.34 | 221.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 228.99 | 223.26 | 222.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 230.80 | 231.33 | 229.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:15:00 | 228.62 | 231.33 | 229.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 227.90 | 230.64 | 228.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 227.90 | 230.64 | 228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 229.96 | 230.51 | 228.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 231.10 | 230.80 | 229.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 228.40 | 229.73 | 229.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 228.40 | 229.73 | 229.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 227.30 | 229.00 | 229.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 226.20 | 225.87 | 226.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 231.70 | 227.04 | 227.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 231.70 | 227.04 | 227.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 232.93 | 227.04 | 227.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 231.25 | 227.88 | 227.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 244.10 | 231.13 | 229.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 265.61 | 269.44 | 262.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 265.61 | 269.44 | 262.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 263.50 | 266.94 | 263.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 263.50 | 266.94 | 263.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 261.90 | 265.93 | 263.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 261.50 | 265.93 | 263.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 261.55 | 265.05 | 262.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 263.89 | 265.05 | 262.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 263.50 | 265.40 | 265.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 263.50 | 265.40 | 265.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 262.70 | 264.98 | 265.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 11:15:00 | 266.99 | 265.38 | 265.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 11:15:00 | 266.99 | 265.38 | 265.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 266.99 | 265.38 | 265.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 266.99 | 265.38 | 265.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 263.86 | 265.08 | 265.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:15:00 | 262.81 | 265.08 | 265.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 249.67 | 253.39 | 257.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 259.80 | 254.24 | 256.90 | SL hit (close>ema200) qty=0.50 sl=254.24 alert=retest2 |

### Cycle 97 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 265.15 | 258.90 | 258.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 269.50 | 261.73 | 260.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 265.80 | 266.52 | 263.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 250.05 | 266.52 | 263.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 244.35 | 262.08 | 262.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 244.35 | 262.08 | 262.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 254.20 | 260.51 | 261.34 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 260.55 | 259.86 | 259.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 262.75 | 260.86 | 260.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 266.85 | 267.34 | 264.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 266.85 | 267.34 | 264.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 265.50 | 266.97 | 265.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 265.45 | 266.97 | 265.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 265.15 | 266.61 | 265.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 265.15 | 266.61 | 265.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 263.60 | 266.00 | 264.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 263.60 | 266.00 | 264.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 263.00 | 265.40 | 264.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 263.90 | 265.40 | 264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 264.50 | 264.77 | 264.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 265.85 | 264.98 | 264.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 265.00 | 265.25 | 264.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 262.00 | 264.60 | 264.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 262.00 | 264.60 | 264.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 11:15:00 | 261.40 | 263.96 | 264.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 253.20 | 251.81 | 254.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 253.20 | 251.81 | 254.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 253.20 | 251.81 | 254.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 256.05 | 251.81 | 254.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 251.95 | 251.84 | 254.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 252.20 | 251.84 | 254.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 242.30 | 243.67 | 247.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 237.80 | 241.07 | 244.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 236.80 | 239.87 | 243.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 225.91 | 229.00 | 230.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 228.60 | 227.47 | 228.85 | SL hit (close>ema200) qty=0.50 sl=227.47 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 232.87 | 229.50 | 229.25 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 225.50 | 229.04 | 229.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 223.53 | 227.94 | 228.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 225.90 | 225.23 | 226.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:30:00 | 225.70 | 225.23 | 226.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 226.30 | 225.63 | 226.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:30:00 | 225.00 | 225.87 | 226.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 224.30 | 225.23 | 226.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 223.34 | 225.08 | 225.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 224.81 | 225.08 | 225.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 223.72 | 224.81 | 225.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:45:00 | 223.28 | 224.12 | 225.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 223.00 | 223.89 | 224.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 223.25 | 223.21 | 224.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 223.15 | 224.63 | 224.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 213.75 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 213.09 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 212.17 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 213.57 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 212.12 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 211.85 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 212.09 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 211.99 | 217.51 | 220.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 12:15:00 | 208.27 | 208.22 | 212.25 | SL hit (close>ema200) qty=0.50 sl=208.22 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 214.00 | 210.19 | 210.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 214.45 | 212.12 | 211.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 217.51 | 217.70 | 216.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 220.59 | 217.70 | 216.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 225.25 | 219.21 | 216.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:45:00 | 227.38 | 220.89 | 217.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 225.52 | 222.62 | 219.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 229.86 | 223.75 | 220.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 228.83 | 230.97 | 231.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 228.83 | 230.97 | 231.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 227.25 | 229.85 | 230.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 228.43 | 227.79 | 228.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 228.43 | 227.79 | 228.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 228.43 | 227.79 | 228.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 225.50 | 227.74 | 228.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 214.22 | 217.68 | 221.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 215.50 | 214.73 | 217.59 | SL hit (close>ema200) qty=0.50 sl=214.73 alert=retest2 |

### Cycle 105 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 209.93 | 205.76 | 205.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 212.09 | 208.33 | 207.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 224.01 | 224.63 | 220.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 224.01 | 224.63 | 220.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 224.60 | 227.16 | 223.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 224.60 | 227.16 | 223.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 220.65 | 225.86 | 223.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 220.87 | 225.86 | 223.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 218.29 | 224.35 | 222.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 218.29 | 224.35 | 222.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 212.30 | 220.11 | 221.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 211.20 | 217.61 | 219.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 216.13 | 215.26 | 217.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 216.13 | 215.26 | 217.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 217.80 | 215.76 | 217.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 219.40 | 215.76 | 217.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 219.90 | 216.59 | 217.93 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 221.79 | 218.96 | 218.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 222.43 | 219.65 | 219.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 223.65 | 224.08 | 222.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 223.65 | 224.08 | 222.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 222.70 | 223.80 | 222.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 222.70 | 223.80 | 222.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 222.61 | 223.42 | 222.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 222.61 | 223.42 | 222.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 223.20 | 223.38 | 222.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 222.64 | 223.38 | 222.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 219.90 | 222.70 | 222.42 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 220.48 | 221.89 | 222.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 219.80 | 221.47 | 221.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 221.56 | 221.36 | 221.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 221.56 | 221.36 | 221.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 221.56 | 221.36 | 221.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 221.56 | 221.36 | 221.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 221.50 | 221.39 | 221.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 251.80 | 221.39 | 221.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 245.85 | 226.28 | 223.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 254.54 | 235.49 | 228.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 269.10 | 272.15 | 267.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:45:00 | 269.66 | 272.15 | 267.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 269.86 | 271.27 | 267.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:30:00 | 267.89 | 271.27 | 267.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 268.05 | 270.37 | 267.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 268.05 | 270.37 | 267.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 264.67 | 269.23 | 267.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 264.67 | 269.23 | 267.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 263.90 | 268.16 | 267.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 266.31 | 268.16 | 267.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 264.59 | 267.21 | 266.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 264.59 | 267.21 | 266.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 267.64 | 267.31 | 267.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:30:00 | 269.47 | 267.80 | 267.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 272.37 | 267.91 | 267.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 11:45:00 | 273.28 | 269.90 | 268.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 267.28 | 269.78 | 270.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 267.28 | 269.78 | 270.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 264.85 | 268.44 | 269.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 266.56 | 265.01 | 267.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 266.56 | 265.01 | 267.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 266.56 | 265.01 | 267.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 266.59 | 265.01 | 267.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 261.15 | 264.24 | 266.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 260.10 | 263.38 | 265.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:45:00 | 260.50 | 262.50 | 265.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 260.51 | 260.82 | 262.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 260.51 | 260.53 | 262.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 259.87 | 259.99 | 261.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 257.87 | 259.61 | 261.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 257.60 | 259.39 | 260.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 257.87 | 259.45 | 260.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 264.33 | 260.54 | 260.97 | SL hit (close>static) qty=1.00 sl=263.00 alert=retest2 |

### Cycle 111 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 264.67 | 261.36 | 261.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 269.07 | 264.88 | 263.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 265.02 | 267.04 | 265.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 265.02 | 267.04 | 265.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 265.02 | 267.04 | 265.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 265.02 | 267.04 | 265.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 265.85 | 266.80 | 265.59 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 263.39 | 264.92 | 265.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 258.85 | 263.71 | 264.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 243.25 | 242.91 | 247.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:45:00 | 244.40 | 242.91 | 247.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 235.40 | 234.53 | 238.43 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 244.40 | 239.99 | 239.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 253.70 | 243.54 | 241.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 244.95 | 245.13 | 242.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 244.95 | 245.13 | 242.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 240.35 | 243.99 | 242.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 240.35 | 243.99 | 242.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 239.30 | 243.05 | 242.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 242.50 | 243.05 | 242.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 246.55 | 243.63 | 242.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 247.60 | 244.85 | 243.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 240.10 | 244.76 | 244.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 240.10 | 244.76 | 244.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 239.00 | 243.24 | 244.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 235.50 | 234.86 | 238.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 236.00 | 234.86 | 238.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 236.75 | 234.44 | 236.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 236.75 | 234.44 | 236.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 236.10 | 234.77 | 236.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 239.65 | 234.77 | 236.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 241.35 | 236.09 | 236.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 242.60 | 236.09 | 236.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 242.25 | 237.32 | 237.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 242.55 | 237.32 | 237.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 241.20 | 238.09 | 237.77 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 233.35 | 237.19 | 237.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 232.80 | 235.78 | 236.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 237.95 | 234.16 | 235.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 237.95 | 234.16 | 235.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 237.95 | 234.16 | 235.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 226.90 | 234.64 | 235.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 238.00 | 229.31 | 228.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 238.00 | 229.31 | 228.73 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 229.30 | 230.73 | 230.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 226.95 | 229.98 | 230.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 234.88 | 226.55 | 227.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 234.88 | 226.55 | 227.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 234.88 | 226.55 | 227.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 237.26 | 226.55 | 227.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 233.61 | 227.96 | 228.42 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 234.45 | 229.26 | 228.96 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 221.20 | 229.04 | 229.23 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 231.29 | 228.53 | 228.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 232.38 | 229.30 | 228.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 232.95 | 233.14 | 231.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 240.14 | 233.14 | 231.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 238.86 | 239.22 | 237.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 238.20 | 239.22 | 237.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 241.56 | 242.57 | 240.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 243.22 | 242.57 | 240.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:15:00 | 252.15 | 244.76 | 242.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 245.80 | 245.81 | 243.29 | SL hit (close<ema200) qty=0.50 sl=245.81 alert=retest1 |

### Cycle 122 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 293.80 | 296.32 | 296.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 292.40 | 295.54 | 296.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 291.70 | 290.60 | 292.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 291.70 | 290.60 | 292.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 291.70 | 290.60 | 292.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 293.80 | 290.60 | 292.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 292.14 | 290.91 | 292.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 293.04 | 290.91 | 292.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 294.16 | 291.74 | 292.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 294.16 | 291.74 | 292.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 293.57 | 292.11 | 292.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 292.83 | 292.11 | 292.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 301.00 | 294.53 | 293.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 301.00 | 294.53 | 293.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 306.60 | 299.45 | 296.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 298.72 | 305.56 | 302.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 298.72 | 305.56 | 302.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 298.72 | 305.56 | 302.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 298.72 | 305.56 | 302.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 300.17 | 304.49 | 302.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 297.80 | 304.49 | 302.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 303.87 | 304.11 | 302.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 303.24 | 304.11 | 302.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 306.50 | 304.77 | 303.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 302.90 | 304.77 | 303.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 309.75 | 316.93 | 313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 309.75 | 316.93 | 313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 308.00 | 315.14 | 313.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 308.00 | 315.14 | 313.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 311.90 | 312.76 | 312.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 311.90 | 312.76 | 312.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 313.80 | 312.97 | 312.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 313.15 | 312.97 | 312.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 315.85 | 313.54 | 312.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 319.25 | 314.51 | 313.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 319.60 | 316.89 | 315.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 332.05 | 317.23 | 315.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 10:45:00 | 243.40 | 2024-05-23 09:15:00 | 267.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-23 09:15:00 | 247.25 | 2024-05-24 09:15:00 | 271.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 13:45:00 | 255.52 | 2024-06-18 09:15:00 | 281.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-25 13:00:00 | 262.55 | 2024-06-27 14:15:00 | 249.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 13:00:00 | 262.55 | 2024-06-28 09:15:00 | 258.31 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2024-06-25 14:45:00 | 261.40 | 2024-07-01 14:15:00 | 248.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 14:45:00 | 261.40 | 2024-07-01 14:15:00 | 257.05 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2024-07-29 11:45:00 | 283.30 | 2024-07-31 09:15:00 | 286.55 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-30 10:00:00 | 281.35 | 2024-07-31 09:15:00 | 286.55 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest1 | 2024-08-05 09:15:00 | 266.45 | 2024-08-06 14:15:00 | 253.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-05 09:15:00 | 266.45 | 2024-08-07 13:15:00 | 262.85 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2024-08-08 09:15:00 | 262.95 | 2024-08-12 09:15:00 | 280.00 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2024-08-08 12:30:00 | 262.55 | 2024-08-12 09:15:00 | 280.00 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2024-08-21 09:15:00 | 276.25 | 2024-08-27 12:15:00 | 272.95 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-23 09:15:00 | 275.10 | 2024-08-27 13:15:00 | 272.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-08-23 14:15:00 | 275.55 | 2024-08-27 13:15:00 | 272.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-08-26 11:00:00 | 274.85 | 2024-08-27 13:15:00 | 272.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-27 09:15:00 | 275.75 | 2024-08-27 13:15:00 | 272.70 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-02 10:15:00 | 264.50 | 2024-09-03 09:15:00 | 270.85 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-09-11 10:45:00 | 252.65 | 2024-09-12 12:15:00 | 240.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:45:00 | 252.65 | 2024-09-12 15:15:00 | 245.45 | STOP_HIT | 0.50 | 2.85% |
| BUY | retest2 | 2024-09-26 11:30:00 | 269.80 | 2024-09-30 09:15:00 | 262.60 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-09-26 14:00:00 | 266.90 | 2024-09-30 09:15:00 | 262.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-26 14:30:00 | 270.20 | 2024-09-30 09:15:00 | 262.60 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-09-27 11:00:00 | 267.00 | 2024-09-30 09:15:00 | 262.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-09-27 14:15:00 | 266.70 | 2024-09-30 09:15:00 | 262.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2024-10-03 09:15:00 | 255.10 | 2024-10-07 09:15:00 | 242.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-03 09:15:00 | 255.10 | 2024-10-08 10:15:00 | 238.00 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2024-10-09 13:15:00 | 242.50 | 2024-10-10 10:15:00 | 247.65 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-10-15 12:15:00 | 241.15 | 2024-10-16 10:15:00 | 245.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-10-15 13:45:00 | 240.35 | 2024-10-16 10:15:00 | 245.80 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-12-03 09:15:00 | 239.39 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-12-03 11:00:00 | 236.56 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2024-12-03 11:45:00 | 236.75 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-12-03 12:45:00 | 236.64 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-12-05 13:30:00 | 238.39 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-12-06 09:15:00 | 238.17 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-12-06 10:15:00 | 238.94 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-12-09 14:30:00 | 238.15 | 2024-12-10 10:15:00 | 237.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-12-16 14:15:00 | 232.45 | 2024-12-18 12:15:00 | 220.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 15:00:00 | 232.58 | 2024-12-18 12:15:00 | 220.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 231.59 | 2024-12-19 09:15:00 | 220.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 232.45 | 2024-12-20 10:15:00 | 220.42 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2024-12-16 15:00:00 | 232.58 | 2024-12-20 10:15:00 | 220.42 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2024-12-17 09:30:00 | 231.59 | 2024-12-20 10:15:00 | 220.42 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2025-01-08 09:15:00 | 205.58 | 2025-01-09 09:15:00 | 210.65 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-01-09 11:00:00 | 208.05 | 2025-01-13 09:15:00 | 197.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:30:00 | 207.98 | 2025-01-13 09:15:00 | 197.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 208.10 | 2025-01-13 09:15:00 | 197.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 206.81 | 2025-01-13 09:15:00 | 196.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 208.05 | 2025-01-13 13:15:00 | 187.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 11:30:00 | 207.98 | 2025-01-13 13:15:00 | 187.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 208.10 | 2025-01-13 13:15:00 | 187.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 206.81 | 2025-01-13 13:15:00 | 186.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 198.60 | 2025-01-27 09:15:00 | 188.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 198.60 | 2025-01-28 09:15:00 | 178.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 191.21 | 2025-02-10 09:15:00 | 181.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 190.81 | 2025-02-10 09:15:00 | 181.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 191.21 | 2025-02-11 09:15:00 | 172.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 190.81 | 2025-02-11 09:15:00 | 171.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 181.55 | 2025-02-11 09:15:00 | 172.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 181.55 | 2025-02-12 09:15:00 | 163.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 11:30:00 | 160.57 | 2025-02-24 14:15:00 | 158.93 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-02-24 12:45:00 | 159.78 | 2025-02-24 14:15:00 | 158.93 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-03-04 11:45:00 | 145.83 | 2025-03-05 09:15:00 | 150.22 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2025-03-24 09:30:00 | 176.72 | 2025-03-26 10:15:00 | 172.02 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest1 | 2025-03-25 11:45:00 | 175.00 | 2025-03-26 10:15:00 | 172.02 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest1 | 2025-03-25 12:30:00 | 174.99 | 2025-03-26 10:15:00 | 172.02 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest1 | 2025-03-25 13:30:00 | 174.98 | 2025-03-26 10:15:00 | 172.02 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-04-01 11:30:00 | 166.44 | 2025-04-01 14:15:00 | 169.70 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest1 | 2025-04-17 09:15:00 | 176.74 | 2025-04-22 14:15:00 | 176.70 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2025-04-17 10:15:00 | 176.03 | 2025-04-22 14:15:00 | 176.70 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-05-06 09:15:00 | 175.42 | 2025-05-07 15:15:00 | 166.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 175.42 | 2025-05-08 09:15:00 | 170.90 | STOP_HIT | 0.50 | 2.58% |
| BUY | retest2 | 2025-05-16 09:15:00 | 177.08 | 2025-05-19 09:15:00 | 194.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 11:45:00 | 215.25 | 2025-06-11 13:15:00 | 209.05 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-06-10 09:15:00 | 215.80 | 2025-06-11 13:15:00 | 209.05 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-10 14:00:00 | 215.37 | 2025-06-11 13:15:00 | 209.05 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-06-10 15:15:00 | 216.00 | 2025-06-11 13:15:00 | 209.05 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-06-18 12:45:00 | 220.30 | 2025-06-23 09:15:00 | 227.64 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-06-18 14:15:00 | 220.16 | 2025-06-23 09:15:00 | 227.64 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-06-19 09:30:00 | 220.50 | 2025-06-23 09:15:00 | 227.64 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-06-19 10:00:00 | 219.45 | 2025-06-23 09:15:00 | 227.64 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-07-11 10:30:00 | 221.00 | 2025-07-16 13:15:00 | 222.98 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-11 11:15:00 | 221.40 | 2025-07-16 13:15:00 | 222.98 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-11 11:45:00 | 221.50 | 2025-07-16 13:15:00 | 222.98 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-11 14:15:00 | 221.30 | 2025-07-16 13:15:00 | 222.98 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-22 11:00:00 | 217.75 | 2025-07-23 10:15:00 | 222.79 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-31 09:45:00 | 215.00 | 2025-07-31 15:15:00 | 216.35 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-08-05 10:45:00 | 211.11 | 2025-08-08 15:15:00 | 200.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 10:45:00 | 211.11 | 2025-08-11 13:15:00 | 204.20 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-08-28 12:30:00 | 214.22 | 2025-08-29 11:15:00 | 214.68 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-09-18 15:00:00 | 219.10 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2025-09-19 12:00:00 | 218.84 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-09-19 13:30:00 | 218.54 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-09-19 14:30:00 | 218.78 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2025-09-23 14:15:00 | 223.68 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-09-23 15:15:00 | 223.66 | 2025-09-26 13:15:00 | 224.84 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-10-15 11:45:00 | 231.10 | 2025-10-16 15:15:00 | 228.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-29 09:15:00 | 263.89 | 2025-10-30 15:15:00 | 263.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-10-31 13:15:00 | 262.81 | 2025-11-04 14:15:00 | 249.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 13:15:00 | 262.81 | 2025-11-06 09:15:00 | 259.80 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-11-06 09:30:00 | 263.15 | 2025-11-06 12:15:00 | 265.15 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-11-14 12:00:00 | 265.85 | 2025-11-17 10:15:00 | 262.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-17 10:15:00 | 265.00 | 2025-11-17 10:15:00 | 262.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-24 14:15:00 | 237.80 | 2025-12-03 09:15:00 | 225.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 14:15:00 | 237.80 | 2025-12-03 14:15:00 | 228.60 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2025-11-25 09:45:00 | 236.80 | 2025-12-05 09:15:00 | 232.87 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-12-10 11:30:00 | 225.00 | 2025-12-17 09:15:00 | 213.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 15:15:00 | 224.30 | 2025-12-17 09:15:00 | 213.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 09:30:00 | 223.34 | 2025-12-17 09:15:00 | 212.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 10:15:00 | 224.81 | 2025-12-17 09:15:00 | 213.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 13:45:00 | 223.28 | 2025-12-17 09:15:00 | 212.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:30:00 | 223.00 | 2025-12-17 09:15:00 | 211.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 223.25 | 2025-12-17 09:15:00 | 212.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 09:15:00 | 223.15 | 2025-12-17 09:15:00 | 211.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 11:30:00 | 225.00 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 7.44% |
| SELL | retest2 | 2025-12-10 15:15:00 | 224.30 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 7.15% |
| SELL | retest2 | 2025-12-11 09:30:00 | 223.34 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 6.75% |
| SELL | retest2 | 2025-12-11 10:15:00 | 224.81 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 7.36% |
| SELL | retest2 | 2025-12-11 13:45:00 | 223.28 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 6.72% |
| SELL | retest2 | 2025-12-11 14:30:00 | 223.00 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 6.61% |
| SELL | retest2 | 2025-12-12 11:45:00 | 223.25 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 6.71% |
| SELL | retest2 | 2025-12-15 09:15:00 | 223.15 | 2025-12-18 12:15:00 | 208.27 | STOP_HIT | 0.50 | 6.67% |
| BUY | retest2 | 2025-12-26 10:45:00 | 227.38 | 2026-01-05 14:15:00 | 228.83 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-12-26 13:15:00 | 225.52 | 2026-01-05 14:15:00 | 228.83 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-12-29 09:15:00 | 229.86 | 2026-01-05 14:15:00 | 228.83 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-08 11:00:00 | 225.50 | 2026-01-09 14:15:00 | 214.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 225.50 | 2026-01-12 14:15:00 | 215.50 | STOP_HIT | 0.50 | 4.43% |
| BUY | retest2 | 2026-02-16 14:30:00 | 269.47 | 2026-02-19 09:15:00 | 267.28 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-02-17 09:15:00 | 272.37 | 2026-02-19 09:15:00 | 267.28 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-17 11:45:00 | 273.28 | 2026-02-19 09:15:00 | 267.28 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-02-20 11:30:00 | 260.10 | 2026-02-25 09:15:00 | 264.33 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-20 12:45:00 | 260.50 | 2026-02-25 09:15:00 | 264.33 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-23 11:45:00 | 260.51 | 2026-02-25 09:15:00 | 264.33 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-23 12:30:00 | 260.51 | 2026-02-25 10:15:00 | 264.67 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-02-24 12:15:00 | 257.87 | 2026-02-25 10:15:00 | 264.67 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-24 13:15:00 | 257.60 | 2026-02-25 10:15:00 | 264.67 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-02-24 14:30:00 | 257.87 | 2026-02-25 10:15:00 | 264.67 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-03-12 11:30:00 | 247.60 | 2026-03-13 12:15:00 | 240.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2026-03-23 09:15:00 | 226.90 | 2026-03-25 09:15:00 | 238.00 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest1 | 2026-04-08 09:15:00 | 240.14 | 2026-04-13 11:15:00 | 252.15 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 240.14 | 2026-04-13 14:15:00 | 245.80 | STOP_HIT | 0.50 | 2.36% |
| BUY | retest2 | 2026-04-13 10:15:00 | 243.22 | 2026-04-16 12:15:00 | 267.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 14:15:00 | 292.83 | 2026-04-28 10:15:00 | 301.00 | STOP_HIT | 1.00 | -2.79% |
