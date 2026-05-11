# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 217 |
| ALERT1 | 145 |
| ALERT2 | 141 |
| ALERT2_SKIP | 75 |
| ALERT3 | 355 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 170 |
| PARTIAL | 15 |
| TARGET_HIT | 15 |
| STOP_HIT | 160 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 189 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 57 / 132
- **Target hits / Stop hits / Partials:** 14 / 160 / 15
- **Avg / median % per leg:** 0.19% / -0.92%
- **Sum % (uncompounded):** 36.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 21 | 24.7% | 13 | 72 | 0 | 0.28% | 24.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.47% | -6.9% |
| BUY @ 3rd Alert (retest2) | 83 | 21 | 25.3% | 13 | 70 | 0 | 0.37% | 31.1% |
| SELL (all) | 104 | 36 | 34.6% | 1 | 88 | 15 | 0.11% | 11.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| SELL @ 3rd Alert (retest2) | 101 | 36 | 35.6% | 1 | 85 | 15 | 0.16% | 16.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.19% | -11.0% |
| retest2 (combined) | 184 | 57 | 31.0% | 14 | 155 | 15 | 0.26% | 47.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 14:15:00 | 244.88 | 247.16 | 247.36 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 249.60 | 247.65 | 247.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 251.74 | 249.00 | 248.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 241.44 | 248.37 | 248.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 241.44 | 248.37 | 248.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 241.44 | 248.37 | 248.21 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 10:15:00 | 234.69 | 245.63 | 246.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 15:15:00 | 233.38 | 238.41 | 242.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 13:15:00 | 231.00 | 230.34 | 233.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-25 14:00:00 | 231.00 | 230.34 | 233.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 230.59 | 230.38 | 232.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 10:15:00 | 230.19 | 230.38 | 232.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 12:15:00 | 230.24 | 230.59 | 232.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 10:00:00 | 230.15 | 227.02 | 228.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 13:15:00 | 231.30 | 229.10 | 228.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 13:15:00 | 231.30 | 229.10 | 228.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 232.69 | 230.40 | 229.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 257.31 | 258.67 | 253.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 09:30:00 | 256.54 | 258.67 | 253.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 271.54 | 271.43 | 268.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 270.59 | 271.43 | 268.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 334.09 | 336.35 | 330.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:45:00 | 332.98 | 336.35 | 330.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 331.40 | 335.36 | 330.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 329.37 | 333.73 | 329.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 331.00 | 333.18 | 330.04 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 322.79 | 328.03 | 328.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 314.20 | 323.37 | 325.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 319.70 | 316.38 | 320.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 319.70 | 316.38 | 320.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 319.70 | 316.38 | 320.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:30:00 | 321.60 | 316.38 | 320.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 321.69 | 317.44 | 320.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 321.69 | 317.44 | 320.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 320.89 | 318.13 | 320.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 12:45:00 | 318.39 | 318.05 | 320.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 13:15:00 | 326.84 | 319.81 | 320.80 | SL hit (close>static) qty=1.00 sl=322.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 324.20 | 321.85 | 321.59 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 11:15:00 | 320.48 | 321.86 | 322.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 13:15:00 | 319.19 | 321.09 | 321.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 327.38 | 321.67 | 321.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 327.38 | 321.67 | 321.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 327.38 | 321.67 | 321.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:30:00 | 330.40 | 321.67 | 321.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 325.59 | 322.45 | 322.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 328.44 | 325.73 | 324.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 15:15:00 | 326.00 | 326.43 | 325.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 09:15:00 | 324.39 | 326.43 | 325.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 318.69 | 324.88 | 324.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:30:00 | 319.94 | 324.88 | 324.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 10:15:00 | 319.97 | 323.90 | 324.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 14:15:00 | 316.40 | 320.61 | 322.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 323.60 | 320.47 | 321.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 323.60 | 320.47 | 321.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 323.60 | 320.47 | 321.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:00:00 | 323.60 | 320.47 | 321.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 321.24 | 320.62 | 321.92 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 13:15:00 | 327.24 | 321.65 | 321.52 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 318.14 | 320.94 | 321.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 317.02 | 319.69 | 320.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 314.26 | 312.85 | 315.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 314.26 | 312.85 | 315.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 314.26 | 312.85 | 315.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:30:00 | 315.33 | 312.85 | 315.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 314.00 | 313.08 | 315.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 314.00 | 313.08 | 315.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 315.20 | 313.09 | 314.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 315.20 | 313.09 | 314.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 314.40 | 313.36 | 314.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 319.59 | 313.36 | 314.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 317.45 | 314.17 | 314.80 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 12:15:00 | 317.10 | 315.31 | 315.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 13:15:00 | 318.60 | 315.97 | 315.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 14:15:00 | 315.68 | 315.91 | 315.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 15:00:00 | 315.68 | 315.91 | 315.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 314.20 | 315.57 | 315.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 317.27 | 315.57 | 315.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 10:00:00 | 316.77 | 315.81 | 315.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:15:00 | 316.00 | 315.70 | 315.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 11:15:00 | 313.60 | 315.28 | 315.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 11:15:00 | 313.60 | 315.28 | 315.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 311.42 | 314.30 | 314.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 320.81 | 313.86 | 314.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 320.81 | 313.86 | 314.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 320.81 | 313.86 | 314.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 324.80 | 313.86 | 314.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 322.56 | 315.60 | 315.12 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 10:15:00 | 313.75 | 317.18 | 317.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 11:15:00 | 311.64 | 316.07 | 317.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 323.20 | 316.30 | 316.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 09:15:00 | 323.20 | 316.30 | 316.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 323.20 | 316.30 | 316.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 09:45:00 | 324.99 | 316.30 | 316.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 10:15:00 | 325.30 | 318.10 | 317.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 11:15:00 | 329.48 | 320.37 | 318.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 10:15:00 | 323.60 | 323.89 | 321.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 11:00:00 | 323.60 | 323.89 | 321.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 321.80 | 323.47 | 321.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 321.80 | 323.47 | 321.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 321.94 | 323.17 | 321.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:30:00 | 322.00 | 323.17 | 321.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 320.80 | 322.69 | 321.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:45:00 | 321.32 | 322.69 | 321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 318.80 | 321.92 | 321.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:00:00 | 318.80 | 321.92 | 321.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 318.80 | 321.29 | 321.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 09:15:00 | 321.40 | 321.29 | 321.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-31 09:15:00 | 353.54 | 340.08 | 337.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 346.71 | 351.32 | 351.89 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 357.39 | 351.86 | 351.44 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 346.42 | 352.42 | 353.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 342.80 | 347.35 | 348.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 09:15:00 | 341.40 | 340.15 | 343.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 10:15:00 | 341.61 | 340.44 | 342.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 341.61 | 340.44 | 342.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:45:00 | 343.20 | 340.44 | 342.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 341.60 | 340.68 | 342.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:45:00 | 342.37 | 340.68 | 342.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 341.95 | 340.93 | 342.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:30:00 | 342.85 | 340.93 | 342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 345.19 | 341.78 | 342.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:45:00 | 344.20 | 341.78 | 342.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 345.68 | 342.56 | 343.21 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 348.31 | 344.14 | 343.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 13:15:00 | 350.11 | 346.87 | 345.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 10:15:00 | 347.46 | 347.93 | 346.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 11:00:00 | 347.46 | 347.93 | 346.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 344.40 | 347.22 | 346.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 11:45:00 | 344.75 | 347.22 | 346.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 344.98 | 346.77 | 346.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:00:00 | 344.98 | 346.77 | 346.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 346.14 | 346.65 | 346.13 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 15:15:00 | 343.40 | 345.73 | 345.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 340.10 | 343.75 | 344.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 350.66 | 344.08 | 344.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 350.66 | 344.08 | 344.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 350.66 | 344.08 | 344.49 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 354.20 | 346.10 | 345.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 360.60 | 356.05 | 351.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 355.40 | 356.51 | 353.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:45:00 | 355.55 | 356.51 | 353.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 357.60 | 358.57 | 356.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 357.60 | 358.57 | 356.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 356.41 | 358.14 | 356.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:15:00 | 354.91 | 358.14 | 356.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 353.00 | 357.11 | 355.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:30:00 | 353.22 | 357.11 | 355.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 352.10 | 354.71 | 354.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 346.38 | 351.63 | 353.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 11:15:00 | 352.87 | 347.49 | 349.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 11:15:00 | 352.87 | 347.49 | 349.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 352.87 | 347.49 | 349.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:00:00 | 352.87 | 347.49 | 349.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 350.61 | 348.11 | 349.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 15:15:00 | 350.40 | 349.26 | 349.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 09:15:00 | 352.82 | 350.15 | 350.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 352.82 | 350.15 | 350.07 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 348.14 | 350.80 | 350.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 345.90 | 348.93 | 350.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 14:15:00 | 350.87 | 348.80 | 349.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 14:15:00 | 350.87 | 348.80 | 349.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 350.87 | 348.80 | 349.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 350.87 | 348.80 | 349.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 351.13 | 349.27 | 349.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 351.69 | 349.27 | 349.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 349.60 | 349.87 | 350.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 15:00:00 | 349.60 | 349.87 | 350.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 353.00 | 350.23 | 350.14 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 15:15:00 | 348.48 | 350.44 | 350.46 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 353.69 | 351.09 | 350.76 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 12:15:00 | 349.79 | 350.87 | 350.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 13:15:00 | 349.09 | 350.52 | 350.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 351.67 | 350.30 | 350.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 351.67 | 350.30 | 350.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 351.67 | 350.30 | 350.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:45:00 | 353.49 | 350.30 | 350.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 350.42 | 350.33 | 350.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 351.21 | 350.33 | 350.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 351.48 | 350.56 | 350.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:00:00 | 351.48 | 350.56 | 350.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 350.60 | 350.57 | 350.64 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 351.98 | 350.85 | 350.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 374.00 | 355.79 | 353.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 14:15:00 | 364.20 | 365.12 | 359.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 15:00:00 | 364.20 | 365.12 | 359.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 358.30 | 362.91 | 360.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 12:00:00 | 358.30 | 362.91 | 360.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 360.83 | 362.49 | 360.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 12:45:00 | 359.30 | 362.49 | 360.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 360.00 | 361.99 | 360.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 14:00:00 | 360.00 | 361.99 | 360.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 14:15:00 | 360.11 | 361.62 | 360.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 15:00:00 | 360.11 | 361.62 | 360.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 358.80 | 361.05 | 360.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:15:00 | 353.47 | 361.05 | 360.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 345.05 | 357.85 | 358.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 342.23 | 348.77 | 353.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 347.69 | 345.04 | 349.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:00:00 | 347.69 | 345.04 | 349.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 351.18 | 346.27 | 349.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:45:00 | 352.23 | 346.27 | 349.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 350.00 | 347.02 | 349.95 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 350.64 | 350.04 | 350.00 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 13:15:00 | 348.21 | 349.64 | 349.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 346.00 | 347.98 | 348.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 09:15:00 | 353.53 | 348.20 | 348.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 353.53 | 348.20 | 348.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 353.53 | 348.20 | 348.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 352.80 | 348.20 | 348.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 350.80 | 348.72 | 348.83 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 11:15:00 | 350.54 | 349.08 | 348.99 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 347.79 | 348.79 | 348.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 346.17 | 348.27 | 348.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 343.57 | 343.33 | 344.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 343.57 | 343.33 | 344.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 343.57 | 343.33 | 344.91 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 348.39 | 345.78 | 345.45 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 344.40 | 345.33 | 345.38 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 346.67 | 345.60 | 345.50 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 344.73 | 345.93 | 345.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 343.48 | 345.44 | 345.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 344.40 | 343.87 | 344.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 344.40 | 343.87 | 344.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 344.40 | 343.87 | 344.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 10:45:00 | 343.02 | 343.75 | 344.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 12:00:00 | 343.00 | 343.60 | 344.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 12:15:00 | 346.42 | 344.16 | 344.62 | SL hit (close>static) qty=1.00 sl=346.23 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 348.39 | 345.39 | 345.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 350.30 | 346.37 | 345.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 14:15:00 | 346.60 | 347.45 | 346.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 14:15:00 | 346.60 | 347.45 | 346.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 346.60 | 347.45 | 346.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 15:00:00 | 346.60 | 347.45 | 346.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 352.86 | 348.52 | 347.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 10:15:00 | 356.21 | 348.52 | 347.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 15:15:00 | 354.00 | 355.24 | 353.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 10:00:00 | 353.98 | 354.79 | 353.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 11:00:00 | 354.44 | 354.72 | 353.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 352.61 | 354.30 | 353.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 12:00:00 | 352.61 | 354.30 | 353.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 359.19 | 355.27 | 354.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 12:45:00 | 357.40 | 355.27 | 354.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 349.71 | 354.36 | 354.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-09 10:15:00 | 351.19 | 353.72 | 353.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 351.19 | 353.72 | 353.82 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 354.07 | 353.74 | 353.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 11:15:00 | 355.45 | 354.08 | 353.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 14:15:00 | 354.50 | 354.56 | 354.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-10 15:00:00 | 354.50 | 354.56 | 354.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 354.40 | 354.53 | 354.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 09:15:00 | 357.50 | 354.53 | 354.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 10:30:00 | 355.21 | 355.23 | 354.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 13:00:00 | 356.00 | 355.54 | 354.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 14:15:00 | 352.73 | 354.95 | 354.70 | SL hit (close<static) qty=1.00 sl=354.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 15:15:00 | 352.02 | 354.37 | 354.46 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 09:15:00 | 367.60 | 357.01 | 355.65 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 12:15:00 | 357.09 | 360.62 | 360.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 13:15:00 | 356.20 | 358.10 | 359.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 10:15:00 | 352.28 | 351.57 | 354.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 11:00:00 | 352.28 | 351.57 | 354.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 355.60 | 350.88 | 352.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:00:00 | 355.60 | 350.88 | 352.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 351.28 | 350.96 | 352.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:15:00 | 348.52 | 350.96 | 352.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 331.09 | 336.93 | 342.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-25 09:15:00 | 313.67 | 330.91 | 338.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 46 — BUY (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 11:15:00 | 319.74 | 316.89 | 316.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 12:15:00 | 324.62 | 318.43 | 317.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 09:15:00 | 332.50 | 332.59 | 329.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 10:00:00 | 332.50 | 332.59 | 329.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 330.26 | 332.03 | 330.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:00:00 | 330.26 | 332.03 | 330.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 330.66 | 331.75 | 330.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:45:00 | 329.98 | 331.75 | 330.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 330.60 | 331.52 | 330.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 333.35 | 331.52 | 330.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 12:45:00 | 330.80 | 331.14 | 330.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 15:00:00 | 331.00 | 330.94 | 330.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 09:15:00 | 326.92 | 329.73 | 329.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 09:15:00 | 326.92 | 329.73 | 329.98 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 10:15:00 | 332.34 | 330.19 | 329.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 11:15:00 | 333.40 | 330.83 | 330.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 331.45 | 331.69 | 330.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 15:15:00 | 331.45 | 331.69 | 330.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 331.45 | 331.69 | 330.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 322.97 | 331.69 | 330.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 323.50 | 330.05 | 330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 316.00 | 321.02 | 324.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 12:15:00 | 316.82 | 315.37 | 318.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 12:15:00 | 316.82 | 315.37 | 318.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 12:15:00 | 316.82 | 315.37 | 318.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 12:45:00 | 317.62 | 315.37 | 318.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 315.90 | 315.73 | 317.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 10:45:00 | 313.20 | 315.42 | 317.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 09:30:00 | 314.53 | 314.35 | 315.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 10:30:00 | 315.34 | 314.74 | 315.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 325.18 | 318.19 | 317.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 325.18 | 318.19 | 317.26 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 14:15:00 | 317.54 | 319.11 | 319.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 316.31 | 317.76 | 318.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 316.80 | 316.64 | 317.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 316.80 | 316.64 | 317.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 316.80 | 316.64 | 317.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:15:00 | 318.40 | 316.64 | 317.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 317.68 | 316.85 | 317.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 11:00:00 | 317.68 | 316.85 | 317.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 319.80 | 317.44 | 317.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 12:00:00 | 319.80 | 317.44 | 317.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 317.20 | 317.39 | 317.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:45:00 | 316.70 | 317.30 | 317.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:15:00 | 316.65 | 317.30 | 317.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 10:15:00 | 316.01 | 317.12 | 317.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 09:15:00 | 326.80 | 316.12 | 315.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 326.80 | 316.12 | 315.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 329.45 | 319.88 | 317.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 13:15:00 | 350.28 | 350.36 | 343.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 14:00:00 | 350.28 | 350.36 | 343.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 344.13 | 348.97 | 344.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:00:00 | 344.13 | 348.97 | 344.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 345.90 | 348.35 | 344.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 10:30:00 | 348.16 | 345.08 | 344.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 12:15:00 | 347.00 | 345.31 | 344.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 15:15:00 | 346.76 | 347.30 | 345.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 343.53 | 346.55 | 346.08 | SL hit (close<static) qty=1.00 sl=344.01 alert=retest2 |

### Cycle 53 — SELL (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 14:15:00 | 343.95 | 345.65 | 345.73 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 349.03 | 346.20 | 345.93 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 344.48 | 346.46 | 346.62 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 350.36 | 346.23 | 345.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 358.47 | 349.61 | 347.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 361.63 | 362.68 | 358.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 14:45:00 | 361.22 | 362.68 | 358.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 360.20 | 362.18 | 358.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 364.11 | 362.18 | 358.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 10:45:00 | 366.18 | 362.05 | 359.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 349.08 | 362.81 | 362.42 | SL hit (close<static) qty=1.00 sl=358.79 alert=retest2 |

### Cycle 57 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 343.80 | 359.01 | 360.73 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 12:15:00 | 356.46 | 354.63 | 354.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 13:15:00 | 357.38 | 355.18 | 354.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 10:15:00 | 382.66 | 382.68 | 376.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 11:00:00 | 382.66 | 382.68 | 376.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 381.47 | 383.91 | 381.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 380.99 | 383.91 | 381.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 376.81 | 382.49 | 381.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 374.87 | 382.49 | 381.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 377.42 | 381.48 | 380.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:15:00 | 378.24 | 381.48 | 380.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 13:15:00 | 377.29 | 379.92 | 380.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 13:15:00 | 377.29 | 379.92 | 380.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 15:15:00 | 376.96 | 379.05 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 376.32 | 375.10 | 376.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 376.32 | 375.10 | 376.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 376.32 | 375.10 | 376.72 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 378.57 | 377.24 | 377.11 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 14:15:00 | 373.88 | 376.79 | 377.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 371.49 | 375.28 | 376.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 11:15:00 | 373.26 | 365.89 | 368.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 11:15:00 | 373.26 | 365.89 | 368.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 373.26 | 365.89 | 368.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:00:00 | 373.26 | 365.89 | 368.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 369.32 | 366.57 | 368.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 15:15:00 | 368.05 | 367.51 | 368.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 10:15:00 | 371.39 | 369.16 | 369.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 10:15:00 | 371.39 | 369.16 | 369.13 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-01-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 14:15:00 | 367.99 | 369.14 | 369.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 15:15:00 | 366.00 | 368.51 | 368.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 368.73 | 368.56 | 368.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 368.73 | 368.56 | 368.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 368.73 | 368.56 | 368.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 10:15:00 | 369.59 | 368.56 | 368.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 371.61 | 369.17 | 369.11 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 368.25 | 369.70 | 369.75 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 09:15:00 | 372.60 | 370.04 | 369.88 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 364.44 | 369.20 | 369.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 358.80 | 365.77 | 367.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 12:15:00 | 361.41 | 361.07 | 363.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 12:45:00 | 361.82 | 361.07 | 363.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 361.09 | 361.01 | 362.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 15:00:00 | 361.09 | 361.01 | 362.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 368.04 | 362.60 | 363.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 10:00:00 | 368.04 | 362.60 | 363.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 365.28 | 363.14 | 363.35 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 366.60 | 363.83 | 363.65 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 360.32 | 363.68 | 363.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 357.20 | 362.38 | 363.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 10:15:00 | 349.00 | 347.79 | 350.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 349.00 | 347.79 | 350.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 347.50 | 346.87 | 348.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:30:00 | 349.40 | 346.87 | 348.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 348.60 | 347.58 | 348.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 14:15:00 | 345.65 | 347.58 | 348.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 15:00:00 | 346.16 | 347.29 | 348.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 09:45:00 | 346.00 | 346.73 | 347.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 10:15:00 | 350.00 | 347.39 | 348.07 | SL hit (close>static) qty=1.00 sl=348.99 alert=retest2 |

### Cycle 70 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 351.21 | 348.63 | 348.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 15:15:00 | 352.50 | 349.40 | 348.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 12:15:00 | 349.38 | 350.14 | 349.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 12:15:00 | 349.38 | 350.14 | 349.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 349.38 | 350.14 | 349.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:00:00 | 349.38 | 350.14 | 349.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 349.00 | 349.91 | 349.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 348.41 | 349.91 | 349.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 356.39 | 351.21 | 350.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 15:15:00 | 359.19 | 351.21 | 350.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 13:45:00 | 357.65 | 356.25 | 353.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 360.57 | 356.73 | 354.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-06 12:15:00 | 395.11 | 380.13 | 371.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 375.86 | 383.20 | 383.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 367.44 | 376.55 | 380.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 09:15:00 | 334.00 | 329.46 | 338.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-15 09:30:00 | 335.20 | 329.46 | 338.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 338.41 | 332.73 | 337.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:45:00 | 338.59 | 332.73 | 337.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 337.23 | 333.63 | 337.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 13:30:00 | 338.01 | 333.63 | 337.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 338.20 | 334.54 | 337.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 15:00:00 | 338.20 | 334.54 | 337.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 339.58 | 335.55 | 338.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:15:00 | 343.59 | 335.55 | 338.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 339.94 | 336.43 | 338.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:30:00 | 343.97 | 336.43 | 338.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 346.80 | 338.50 | 339.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:45:00 | 348.02 | 338.50 | 339.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 341.78 | 339.16 | 339.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 12:15:00 | 340.05 | 339.16 | 339.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 12:15:00 | 340.91 | 339.51 | 339.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 340.91 | 339.51 | 339.41 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 337.00 | 339.15 | 339.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 15:15:00 | 336.20 | 338.04 | 338.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 345.33 | 333.31 | 334.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 345.33 | 333.31 | 334.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 345.33 | 333.31 | 334.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 345.33 | 333.31 | 334.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 10:15:00 | 344.80 | 335.61 | 335.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 15:15:00 | 348.40 | 341.74 | 338.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 341.60 | 344.48 | 342.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 341.60 | 344.48 | 342.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 341.60 | 344.48 | 342.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 341.01 | 344.48 | 342.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 337.28 | 343.04 | 341.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:00:00 | 337.28 | 343.04 | 341.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 335.12 | 341.46 | 341.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 12:00:00 | 335.12 | 341.46 | 341.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 337.21 | 340.61 | 340.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 334.01 | 336.57 | 338.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 336.59 | 336.57 | 338.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 15:00:00 | 336.59 | 336.57 | 338.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 326.08 | 326.79 | 329.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:45:00 | 327.50 | 326.79 | 329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 329.06 | 326.36 | 328.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 331.06 | 326.36 | 328.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 328.99 | 326.88 | 328.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:45:00 | 329.33 | 326.88 | 328.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 329.00 | 327.31 | 328.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 329.00 | 327.31 | 328.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 329.40 | 327.73 | 328.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 13:15:00 | 328.06 | 327.73 | 328.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 14:15:00 | 328.21 | 327.99 | 328.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 12:15:00 | 331.00 | 329.61 | 329.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 331.00 | 329.61 | 329.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 344.60 | 332.61 | 330.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 335.19 | 336.72 | 334.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:00:00 | 335.19 | 336.72 | 334.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 338.07 | 336.99 | 334.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 14:15:00 | 340.74 | 337.61 | 335.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 332.31 | 337.03 | 335.89 | SL hit (close<static) qty=1.00 sl=333.80 alert=retest2 |

### Cycle 77 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 325.98 | 333.55 | 334.43 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 342.68 | 335.55 | 334.58 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 331.33 | 335.50 | 335.50 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 09:15:00 | 345.80 | 337.19 | 336.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 12:15:00 | 349.79 | 341.75 | 338.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 352.46 | 353.32 | 347.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 12:00:00 | 352.46 | 353.32 | 347.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 345.65 | 351.04 | 347.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:30:00 | 341.93 | 351.04 | 347.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 347.98 | 350.43 | 347.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 351.99 | 350.43 | 347.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 355.58 | 351.46 | 348.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 363.33 | 355.53 | 351.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 10:30:00 | 361.38 | 361.41 | 356.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 362.06 | 366.17 | 366.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 362.06 | 366.17 | 366.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 358.85 | 364.04 | 365.38 | Break + close below crossover candle low |

### Cycle 82 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 383.20 | 365.36 | 364.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 402.89 | 381.74 | 374.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 381.47 | 385.47 | 379.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 381.47 | 385.47 | 379.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 379.20 | 384.22 | 379.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 378.00 | 384.22 | 379.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 373.20 | 382.01 | 378.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 372.40 | 382.01 | 378.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 378.65 | 381.34 | 378.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:30:00 | 383.00 | 382.15 | 379.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 15:15:00 | 379.00 | 381.85 | 380.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 11:15:00 | 376.06 | 379.19 | 379.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 376.06 | 379.19 | 379.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 12:15:00 | 374.50 | 378.25 | 378.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 373.82 | 371.49 | 373.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 373.82 | 371.49 | 373.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 373.82 | 371.49 | 373.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:30:00 | 373.56 | 371.49 | 373.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 382.36 | 373.67 | 374.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:00:00 | 382.36 | 373.67 | 374.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 381.20 | 375.17 | 375.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 385.70 | 377.28 | 376.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 13:15:00 | 407.84 | 408.07 | 399.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 13:45:00 | 407.38 | 408.07 | 399.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 464.39 | 469.32 | 458.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 458.44 | 469.32 | 458.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 490.29 | 483.20 | 477.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 11:15:00 | 498.98 | 484.32 | 478.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:15:00 | 495.99 | 487.90 | 485.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 499.28 | 490.05 | 486.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:30:00 | 494.56 | 491.17 | 487.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 12:15:00 | 496.80 | 498.71 | 496.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 12:45:00 | 496.20 | 498.71 | 496.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 492.39 | 497.44 | 496.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 13:45:00 | 490.20 | 497.44 | 496.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-24 14:15:00 | 485.05 | 494.96 | 495.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 14:15:00 | 485.05 | 494.96 | 495.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 10:15:00 | 482.79 | 485.59 | 488.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 15:15:00 | 469.36 | 466.25 | 470.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 09:15:00 | 473.20 | 466.25 | 470.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 467.64 | 466.53 | 469.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:30:00 | 473.34 | 466.53 | 469.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 482.53 | 469.73 | 471.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 482.53 | 469.73 | 471.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 482.39 | 472.26 | 472.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 486.37 | 479.99 | 476.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 478.07 | 479.79 | 476.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 11:15:00 | 478.07 | 479.79 | 476.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 478.07 | 479.79 | 476.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 478.99 | 479.79 | 476.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 472.42 | 478.31 | 476.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 472.42 | 478.31 | 476.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 484.00 | 479.45 | 477.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:45:00 | 485.97 | 479.60 | 477.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 485.08 | 480.53 | 479.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 15:15:00 | 476.60 | 478.48 | 478.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 476.60 | 478.48 | 478.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 465.11 | 474.24 | 476.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 11:15:00 | 478.80 | 474.79 | 476.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 11:15:00 | 478.80 | 474.79 | 476.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 478.80 | 474.79 | 476.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 478.80 | 474.79 | 476.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 478.53 | 475.53 | 476.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:30:00 | 479.00 | 475.53 | 476.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 477.34 | 476.64 | 476.65 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 480.04 | 477.33 | 476.97 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 476.23 | 477.46 | 477.54 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 478.40 | 477.73 | 477.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 482.00 | 478.61 | 478.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 475.54 | 478.70 | 478.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 474.59 | 477.88 | 477.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 15:15:00 | 472.01 | 476.33 | 477.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 480.94 | 477.10 | 477.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 480.05 | 477.69 | 477.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 477.19 | 477.69 | 477.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 13:15:00 | 483.78 | 478.05 | 477.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 13:15:00 | 483.78 | 478.05 | 477.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 496.05 | 483.43 | 480.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 496.59 | 523.88 | 514.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 492.80 | 517.67 | 512.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 488.01 | 517.67 | 512.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 485.69 | 506.20 | 507.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 445.79 | 490.73 | 500.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 463.30 | 454.34 | 463.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 459.58 | 455.39 | 462.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 458.23 | 455.39 | 462.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:45:00 | 458.21 | 455.47 | 462.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 458.63 | 455.10 | 459.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:15:00 | 458.00 | 450.46 | 451.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 456.20 | 452.82 | 452.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 456.20 | 452.82 | 452.56 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 426.44 | 448.21 | 450.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 402.51 | 439.07 | 446.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 413.40 | 409.19 | 421.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 413.40 | 409.19 | 421.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 426.49 | 413.29 | 420.87 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 427.80 | 424.26 | 424.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 430.19 | 425.45 | 424.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 436.20 | 436.25 | 433.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 438.93 | 436.78 | 433.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 443.82 | 445.49 | 442.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 443.58 | 445.49 | 442.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 442.33 | 444.86 | 442.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 442.33 | 444.86 | 442.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 442.56 | 444.40 | 442.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 443.20 | 444.16 | 442.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:00:00 | 443.18 | 443.96 | 442.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 444.40 | 443.44 | 442.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 439.86 | 443.34 | 442.98 | SL hit (close<static) qty=1.00 sl=440.88 alert=retest2 |

### Cycle 97 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 439.10 | 442.49 | 442.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 438.21 | 440.29 | 441.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 455.66 | 443.53 | 441.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 463.00 | 451.57 | 446.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 15:15:00 | 462.20 | 462.94 | 455.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:15:00 | 466.47 | 462.94 | 455.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 12:15:00 | 472.13 | 462.42 | 457.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 453.00 | 460.94 | 458.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 453.00 | 460.94 | 458.72 | SL hit (close<ema400) qty=1.00 sl=458.72 alert=retest1 |

### Cycle 99 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 452.66 | 456.77 | 457.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 451.96 | 455.81 | 456.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 440.60 | 434.38 | 436.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 440.61 | 435.63 | 436.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:45:00 | 440.97 | 435.63 | 436.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 442.19 | 437.84 | 437.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 447.20 | 441.91 | 439.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 443.42 | 444.59 | 442.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 443.42 | 444.59 | 442.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 442.37 | 444.14 | 442.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 444.70 | 444.14 | 442.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:30:00 | 445.25 | 444.38 | 442.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:30:00 | 446.57 | 450.76 | 450.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 441.88 | 448.99 | 449.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 441.88 | 448.99 | 449.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 440.70 | 444.01 | 446.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 427.60 | 427.37 | 431.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 11:00:00 | 427.60 | 427.37 | 431.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 422.70 | 423.31 | 425.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 427.61 | 423.31 | 425.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 417.40 | 420.06 | 422.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 419.19 | 420.06 | 422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 416.02 | 416.75 | 419.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 416.02 | 416.75 | 419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 407.77 | 410.30 | 412.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 404.20 | 411.34 | 412.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 422.84 | 410.27 | 410.77 | SL hit (close>static) qty=1.00 sl=415.80 alert=retest2 |

### Cycle 102 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 425.78 | 413.37 | 412.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 429.58 | 418.18 | 414.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 424.01 | 424.36 | 420.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:45:00 | 424.80 | 424.13 | 421.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 422.45 | 423.85 | 421.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 421.75 | 423.85 | 421.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 424.22 | 423.93 | 422.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:15:00 | 423.00 | 423.93 | 422.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 423.00 | 423.74 | 422.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 430.91 | 423.74 | 422.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 429.03 | 423.58 | 423.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 436.18 | 440.67 | 440.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 436.18 | 440.67 | 440.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 434.00 | 438.52 | 439.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 423.78 | 418.12 | 423.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 410.00 | 416.50 | 422.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:15:00 | 404.87 | 416.50 | 422.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 409.80 | 408.15 | 413.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 409.94 | 408.50 | 413.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 409.60 | 410.80 | 412.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 412.50 | 410.95 | 412.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 410.60 | 410.95 | 412.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 411.00 | 410.96 | 412.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 418.10 | 413.13 | 412.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 418.10 | 413.13 | 412.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 422.03 | 416.60 | 414.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 402.00 | 420.79 | 421.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 09:15:00 | 397.69 | 406.14 | 412.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 400.98 | 400.39 | 405.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:45:00 | 402.20 | 400.39 | 405.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 405.20 | 401.76 | 404.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 405.20 | 401.76 | 404.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 404.60 | 402.33 | 404.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 405.76 | 402.33 | 404.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 405.74 | 403.01 | 404.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 404.00 | 403.31 | 404.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 404.10 | 404.49 | 404.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 406.32 | 405.18 | 405.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 406.32 | 405.18 | 405.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 406.80 | 405.50 | 405.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 406.03 | 406.37 | 405.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 406.00 | 406.30 | 405.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 405.80 | 406.30 | 405.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 405.80 | 406.20 | 405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 405.70 | 406.20 | 405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 405.98 | 406.15 | 405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 405.98 | 406.15 | 405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 405.88 | 406.10 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 405.88 | 406.10 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 404.31 | 405.74 | 405.80 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 407.88 | 405.95 | 405.88 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 405.44 | 406.14 | 406.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 403.80 | 405.27 | 405.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 397.95 | 396.62 | 398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 399.02 | 396.72 | 397.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 399.34 | 396.72 | 397.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 397.70 | 396.91 | 397.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 395.60 | 396.00 | 397.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:00:00 | 395.78 | 395.68 | 396.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 395.00 | 395.74 | 396.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:30:00 | 395.77 | 395.65 | 396.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 398.63 | 396.18 | 396.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 398.57 | 396.66 | 396.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 398.57 | 396.66 | 396.50 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 394.95 | 396.38 | 396.48 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 12:15:00 | 401.72 | 397.26 | 396.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 408.45 | 401.68 | 399.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 404.40 | 404.99 | 402.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:00:00 | 404.40 | 404.99 | 402.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 402.20 | 404.04 | 402.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 402.20 | 404.04 | 402.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 400.58 | 403.35 | 402.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 400.58 | 403.35 | 402.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 398.52 | 401.60 | 401.81 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 402.69 | 401.54 | 401.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 13:15:00 | 403.20 | 402.08 | 401.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 419.58 | 424.14 | 416.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:45:00 | 418.26 | 424.14 | 416.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 446.65 | 450.49 | 444.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:30:00 | 456.80 | 452.13 | 446.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:30:00 | 454.73 | 452.80 | 447.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:00:00 | 454.14 | 453.07 | 448.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 455.00 | 453.45 | 448.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 452.40 | 452.88 | 449.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 459.94 | 453.20 | 450.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 458.00 | 452.22 | 450.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 457.00 | 452.82 | 450.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 457.40 | 454.60 | 452.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 502.48 | 468.17 | 460.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 480.89 | 486.99 | 487.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 471.00 | 483.79 | 486.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 480.34 | 480.05 | 482.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 480.34 | 480.05 | 482.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 460.48 | 454.33 | 463.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 460.48 | 454.33 | 463.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 467.68 | 457.00 | 463.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 467.60 | 457.00 | 463.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 463.96 | 458.39 | 463.82 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 484.97 | 468.07 | 467.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 490.22 | 472.50 | 469.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 492.21 | 492.86 | 485.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:00:00 | 492.21 | 492.86 | 485.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 487.40 | 490.98 | 486.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 484.40 | 490.98 | 486.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 487.89 | 490.36 | 486.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 483.90 | 490.36 | 486.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 487.58 | 489.81 | 486.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 487.43 | 489.81 | 486.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 495.89 | 491.02 | 487.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 515.94 | 497.15 | 493.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:00:00 | 502.44 | 499.55 | 495.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 502.89 | 499.97 | 496.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 502.44 | 498.51 | 497.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 500.40 | 498.72 | 497.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 14:00:00 | 512.00 | 503.06 | 500.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 495.65 | 500.15 | 500.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 495.65 | 500.15 | 500.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 477.75 | 495.64 | 498.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 456.15 | 445.36 | 460.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 456.15 | 445.36 | 460.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 424.50 | 421.41 | 428.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 423.00 | 422.51 | 427.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 421.90 | 423.41 | 426.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:30:00 | 422.05 | 423.40 | 426.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 422.85 | 423.40 | 426.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 426.25 | 423.96 | 426.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 426.25 | 423.96 | 426.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 427.55 | 424.68 | 426.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 427.55 | 424.68 | 426.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 427.25 | 425.19 | 426.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 429.55 | 425.19 | 426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 430.90 | 426.34 | 426.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 430.90 | 426.34 | 426.77 | SL hit (close>static) qty=1.00 sl=429.15 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 431.30 | 427.33 | 427.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 435.75 | 429.67 | 428.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 431.90 | 430.78 | 429.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 431.90 | 431.00 | 429.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 433.15 | 430.51 | 429.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 426.50 | 433.83 | 431.87 | SL hit (close<static) qty=1.00 sl=429.50 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 439.85 | 443.42 | 443.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 436.60 | 442.06 | 442.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:00:00 | 436.00 | 434.11 | 437.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 417.30 | 411.94 | 414.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 417.30 | 411.94 | 414.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 415.35 | 412.62 | 414.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:30:00 | 414.15 | 414.12 | 414.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 14:15:00 | 418.40 | 414.98 | 414.99 | SL hit (close>static) qty=1.00 sl=417.30 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 15:15:00 | 417.35 | 415.45 | 415.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 423.70 | 417.10 | 415.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 12:15:00 | 417.95 | 418.43 | 416.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 13:00:00 | 417.95 | 418.43 | 416.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 412.60 | 417.27 | 416.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 412.60 | 417.27 | 416.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 410.30 | 415.87 | 416.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 409.25 | 414.55 | 415.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 411.75 | 408.91 | 411.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 408.85 | 408.90 | 410.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 14:30:00 | 407.90 | 408.83 | 410.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 417.00 | 410.42 | 411.00 | SL hit (close>static) qty=1.00 sl=411.75 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 418.25 | 411.99 | 411.66 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 411.30 | 412.66 | 412.78 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 417.70 | 413.53 | 413.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 426.40 | 418.00 | 415.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 14:15:00 | 438.60 | 438.88 | 434.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 14:45:00 | 438.00 | 438.88 | 434.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 477.55 | 446.76 | 439.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 484.95 | 455.19 | 443.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-04 09:15:00 | 533.45 | 496.67 | 472.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 563.10 | 569.56 | 570.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 558.00 | 566.25 | 568.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 571.20 | 555.31 | 560.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 571.80 | 558.61 | 561.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 572.50 | 558.61 | 561.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 571.30 | 563.93 | 563.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 579.10 | 568.54 | 565.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:15:00 | 564.15 | 573.08 | 569.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 566.20 | 571.70 | 569.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 565.75 | 571.70 | 569.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 565.00 | 567.91 | 568.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 15:15:00 | 557.00 | 563.94 | 565.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 567.15 | 562.47 | 564.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 566.70 | 563.31 | 564.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 568.05 | 563.31 | 564.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 573.00 | 565.63 | 565.20 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 542.00 | 560.54 | 563.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 11:15:00 | 537.90 | 546.32 | 554.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:45:00 | 546.55 | 539.87 | 547.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 519.70 | 515.97 | 521.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 519.30 | 515.97 | 521.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 508.30 | 505.47 | 510.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 507.80 | 505.47 | 510.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 510.40 | 506.46 | 510.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:30:00 | 509.65 | 506.46 | 510.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 509.80 | 507.12 | 510.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 510.35 | 507.12 | 510.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 510.40 | 507.78 | 510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 513.75 | 507.78 | 510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 513.15 | 508.85 | 510.87 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 523.10 | 512.57 | 512.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 527.25 | 517.42 | 514.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 530.75 | 530.99 | 526.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 530.75 | 530.99 | 526.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 526.80 | 529.88 | 527.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 516.10 | 529.88 | 527.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 512.55 | 526.42 | 525.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 512.55 | 526.42 | 525.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 502.85 | 521.70 | 523.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 502.05 | 517.77 | 521.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 430.10 | 429.38 | 438.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 440.50 | 429.38 | 438.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 430.85 | 429.68 | 437.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 428.40 | 431.30 | 436.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 427.95 | 432.82 | 434.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 428.00 | 432.00 | 433.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:15:00 | 428.80 | 430.49 | 432.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 425.80 | 429.28 | 431.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 422.45 | 429.83 | 430.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:15:00 | 424.75 | 425.26 | 427.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 406.98 | 417.91 | 423.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 407.36 | 417.91 | 423.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 406.55 | 416.07 | 421.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 406.60 | 416.07 | 421.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 416.65 | 415.31 | 420.53 | SL hit (close>ema200) qty=0.50 sl=415.31 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 376.70 | 376.03 | 375.94 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 373.55 | 375.78 | 375.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 369.15 | 374.59 | 375.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 13:15:00 | 366.20 | 366.15 | 369.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 14:00:00 | 366.20 | 366.15 | 369.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 354.50 | 363.59 | 367.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:45:00 | 350.00 | 357.78 | 363.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:00:00 | 351.05 | 356.43 | 362.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 340.95 | 355.54 | 361.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 347.55 | 346.77 | 351.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 333.50 | 338.68 | 343.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 339.20 | 338.79 | 343.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 339.20 | 338.79 | 343.44 | SL hit (close>ema200) qty=0.50 sl=338.79 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 356.10 | 344.07 | 342.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 360.65 | 353.28 | 348.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 357.90 | 358.95 | 354.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 356.75 | 358.85 | 356.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 356.75 | 358.85 | 356.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 353.65 | 358.85 | 356.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 356.85 | 358.45 | 356.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 359.65 | 358.32 | 356.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 11:15:00 | 349.95 | 355.89 | 356.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 349.95 | 355.89 | 356.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 342.80 | 351.58 | 354.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 346.85 | 346.42 | 349.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:15:00 | 340.60 | 346.42 | 349.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:45:00 | 345.40 | 345.56 | 348.37 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 12:30:00 | 345.40 | 345.57 | 347.89 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 348.40 | 346.14 | 347.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-28 13:15:00 | 348.40 | 346.14 | 347.93 | SL hit (close>ema400) qty=1.00 sl=347.93 alert=retest1 |

### Cycle 136 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 357.00 | 350.46 | 349.72 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 343.80 | 349.13 | 349.18 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 350.20 | 348.86 | 348.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 362.50 | 351.59 | 350.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 390.85 | 391.72 | 385.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:45:00 | 389.35 | 391.72 | 385.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 389.00 | 390.71 | 386.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 401.25 | 390.71 | 386.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 404.60 | 393.49 | 388.43 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 396.65 | 400.00 | 400.34 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 405.45 | 401.10 | 400.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 411.00 | 404.92 | 402.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 432.35 | 432.46 | 427.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 14:30:00 | 431.20 | 432.46 | 427.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 490.70 | 444.12 | 433.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 497.05 | 463.34 | 446.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 502.75 | 479.84 | 460.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 510.95 | 486.76 | 483.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 498.25 | 494.33 | 488.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 477.40 | 493.25 | 491.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 477.40 | 493.25 | 491.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 478.85 | 490.37 | 490.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 476.45 | 487.58 | 489.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 476.45 | 487.58 | 489.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 474.05 | 484.88 | 487.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 454.85 | 454.34 | 466.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 454.85 | 454.34 | 466.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 458.40 | 456.05 | 465.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 456.70 | 456.89 | 465.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 456.50 | 459.53 | 464.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 455.75 | 447.51 | 453.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 466.25 | 456.85 | 455.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 466.25 | 456.85 | 455.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 477.15 | 460.91 | 457.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 478.95 | 481.68 | 476.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 478.95 | 481.68 | 476.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 476.10 | 480.56 | 476.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 477.15 | 480.56 | 476.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 477.20 | 479.89 | 476.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 476.05 | 479.89 | 476.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 481.85 | 480.28 | 476.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 485.70 | 480.69 | 478.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:15:00 | 484.30 | 481.15 | 478.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:00:00 | 484.65 | 482.41 | 480.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 470.70 | 479.29 | 479.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 470.70 | 479.29 | 479.32 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 480.30 | 477.85 | 477.65 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 475.00 | 477.39 | 477.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 461.40 | 474.20 | 476.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 479.05 | 467.96 | 470.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 482.20 | 470.81 | 471.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 481.70 | 470.81 | 471.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 484.60 | 473.57 | 472.83 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 471.00 | 474.34 | 474.68 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 11:15:00 | 475.65 | 474.94 | 474.86 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 465.35 | 473.01 | 474.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 464.10 | 471.22 | 473.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 465.25 | 463.86 | 467.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:00:00 | 465.25 | 463.86 | 467.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 465.85 | 464.26 | 467.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 465.15 | 464.26 | 467.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 466.45 | 464.70 | 467.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 466.45 | 464.70 | 467.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 468.90 | 465.54 | 467.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 468.90 | 465.54 | 467.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 466.65 | 465.76 | 467.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 468.50 | 465.76 | 467.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 467.50 | 466.11 | 467.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 462.25 | 466.11 | 467.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 456.80 | 464.25 | 466.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 454.80 | 461.56 | 464.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:45:00 | 454.95 | 459.67 | 463.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 15:15:00 | 432.06 | 442.95 | 448.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 15:15:00 | 432.20 | 442.95 | 448.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 455.65 | 433.21 | 437.71 | SL hit (close>ema200) qty=0.50 sl=433.21 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 461.35 | 443.13 | 441.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 464.00 | 454.33 | 448.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 498.10 | 519.07 | 507.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 502.30 | 515.72 | 506.81 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 494.60 | 503.13 | 503.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 491.00 | 496.00 | 499.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 509.35 | 488.64 | 492.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 499.00 | 490.72 | 493.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 501.00 | 490.72 | 493.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 499.20 | 494.43 | 494.32 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 491.05 | 493.98 | 494.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 487.50 | 490.43 | 492.11 | Break + close below crossover candle low |

### Cycle 154 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 507.55 | 493.86 | 493.51 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 492.00 | 495.29 | 495.64 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 502.00 | 496.37 | 496.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 517.15 | 500.53 | 497.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 520.00 | 520.88 | 515.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 520.00 | 520.88 | 515.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 514.65 | 520.34 | 517.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 514.65 | 520.34 | 517.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 517.95 | 519.86 | 517.94 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 511.70 | 516.05 | 516.62 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 533.95 | 518.43 | 517.43 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 517.45 | 524.76 | 525.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 14:15:00 | 514.00 | 522.61 | 524.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 521.45 | 521.21 | 523.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 503.95 | 502.16 | 506.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 509.70 | 502.16 | 506.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 504.15 | 502.71 | 505.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 500.20 | 502.33 | 505.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 500.20 | 501.92 | 504.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:45:00 | 500.10 | 501.71 | 504.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 498.00 | 502.16 | 504.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 498.00 | 501.32 | 503.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 502.15 | 501.34 | 503.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 500.45 | 501.16 | 503.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 503.20 | 501.16 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 495.80 | 499.26 | 501.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 498.10 | 499.26 | 501.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 501.30 | 499.67 | 501.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 501.30 | 499.67 | 501.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 497.00 | 499.13 | 501.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 503.50 | 499.13 | 501.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 499.60 | 499.23 | 501.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 503.90 | 499.23 | 501.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 493.00 | 488.83 | 490.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 494.05 | 488.83 | 490.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 493.10 | 489.69 | 491.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 493.10 | 489.69 | 491.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 491.85 | 490.12 | 491.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 499.30 | 490.12 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 512.65 | 500.55 | 497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 510.30 | 511.02 | 506.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 510.80 | 511.02 | 506.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 502.45 | 511.00 | 509.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 502.45 | 511.00 | 509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 503.95 | 509.59 | 508.94 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 502.75 | 508.23 | 508.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 495.85 | 502.61 | 504.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 495.10 | 492.71 | 495.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 494.50 | 493.34 | 495.71 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 499.00 | 496.41 | 496.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 506.65 | 500.19 | 498.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 502.90 | 502.92 | 500.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 495.15 | 502.92 | 500.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 499.55 | 502.24 | 500.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 497.15 | 502.24 | 500.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 518.50 | 505.49 | 502.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 527.25 | 513.03 | 506.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 527.40 | 513.03 | 506.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 531.35 | 535.32 | 537.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 519.95 | 517.65 | 524.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 518.45 | 517.65 | 524.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 515.55 | 516.53 | 520.63 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 528.30 | 522.49 | 521.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 591.85 | 542.18 | 532.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 570.50 | 575.35 | 555.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 570.50 | 575.35 | 555.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 557.60 | 570.92 | 556.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 557.60 | 570.92 | 556.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 556.05 | 567.94 | 556.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 555.70 | 567.94 | 556.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 552.35 | 564.83 | 556.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 552.35 | 564.83 | 556.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 554.35 | 562.73 | 556.12 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 531.70 | 549.40 | 551.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 525.15 | 544.55 | 548.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 520.00 | 518.26 | 523.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 520.00 | 518.26 | 523.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 521.85 | 518.98 | 523.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 516.15 | 518.98 | 523.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 505.00 | 507.80 | 508.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 502.40 | 506.72 | 507.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 501.10 | 504.48 | 505.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 476.05 | 481.83 | 486.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 472.30 | 467.82 | 472.84 | SL hit (close>ema200) qty=0.50 sl=467.82 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 478.45 | 474.33 | 474.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 481.95 | 476.70 | 475.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 478.75 | 478.81 | 476.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 478.75 | 478.81 | 476.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 479.00 | 478.85 | 477.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 484.40 | 478.85 | 477.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 521.30 | 519.63 | 519.54 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 518.20 | 519.35 | 519.42 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 520.20 | 519.52 | 519.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 521.60 | 520.09 | 519.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 519.15 | 519.90 | 519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 519.90 | 519.90 | 519.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 517.50 | 519.90 | 519.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 522.30 | 520.38 | 519.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 527.70 | 521.85 | 520.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 527.30 | 522.88 | 521.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 526.65 | 523.30 | 521.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 526.45 | 523.30 | 521.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 521.65 | 523.23 | 521.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 521.65 | 523.23 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 521.40 | 522.87 | 521.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 521.40 | 522.87 | 521.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 521.80 | 522.65 | 521.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 515.60 | 522.65 | 521.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 506.70 | 515.20 | 518.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 517.55 | 512.63 | 516.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 510.85 | 512.27 | 515.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 508.50 | 511.38 | 514.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 510.00 | 510.65 | 513.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 509.85 | 507.87 | 509.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 514.00 | 511.27 | 510.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 518.00 | 518.68 | 516.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 521.15 | 518.68 | 516.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 514.95 | 517.95 | 516.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 514.10 | 517.95 | 516.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 517.80 | 517.92 | 516.62 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 510.80 | 515.43 | 515.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 510.00 | 514.35 | 515.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 515.55 | 514.48 | 515.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 518.20 | 515.22 | 515.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 522.55 | 515.22 | 515.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 521.20 | 516.42 | 515.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 12:15:00 | 534.40 | 520.01 | 517.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:30:00 | 517.85 | 519.42 | 517.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 517.00 | 518.93 | 517.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 519.45 | 518.93 | 517.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 508.60 | 516.95 | 519.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 517.30 | 514.08 | 516.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 519.45 | 515.15 | 516.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 519.70 | 515.15 | 516.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 520.85 | 516.29 | 517.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 529.60 | 516.29 | 517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 517.90 | 517.49 | 517.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 517.90 | 517.49 | 517.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 518.80 | 517.75 | 517.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 519.50 | 517.75 | 517.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 520.20 | 518.24 | 518.04 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 516.50 | 517.89 | 517.90 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 518.90 | 518.03 | 517.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 520.60 | 518.66 | 518.26 | Break + close above crossover candle high |

### Cycle 181 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 512.20 | 517.46 | 517.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 509.60 | 515.89 | 517.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 520.00 | 516.32 | 516.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 520.10 | 517.08 | 517.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 521.00 | 517.08 | 517.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 521.85 | 518.03 | 517.64 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 514.55 | 517.66 | 517.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 512.50 | 516.63 | 517.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 512.35 | 510.43 | 512.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 513.40 | 511.03 | 512.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 513.40 | 511.03 | 512.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 512.50 | 511.32 | 512.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:15:00 | 511.65 | 511.32 | 512.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 523.95 | 516.22 | 514.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 516.50 | 517.65 | 516.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 515.90 | 517.30 | 516.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 516.45 | 517.30 | 516.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 515.50 | 516.94 | 516.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 518.60 | 516.94 | 516.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 520.65 | 518.36 | 517.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 517.70 | 518.36 | 517.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 585.85 | 586.25 | 582.04 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 567.85 | 579.50 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 564.65 | 576.53 | 579.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 566.75 | 539.69 | 547.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 565.50 | 544.85 | 549.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 565.50 | 544.85 | 549.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 539.50 | 546.47 | 549.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:45:00 | 535.10 | 543.90 | 547.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:15:00 | 508.34 | 516.02 | 520.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 517.70 | 511.94 | 515.18 | SL hit (close>ema200) qty=0.50 sl=511.94 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 519.80 | 516.27 | 516.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 525.80 | 520.00 | 518.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 519.20 | 520.08 | 518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 517.50 | 519.57 | 518.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 517.50 | 519.57 | 518.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 514.70 | 518.59 | 518.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 514.70 | 518.59 | 518.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 514.30 | 517.06 | 517.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 509.20 | 515.49 | 516.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 528.85 | 503.33 | 505.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 519.00 | 506.46 | 506.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 523.00 | 506.46 | 506.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 516.05 | 508.38 | 507.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 522.50 | 512.80 | 509.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 536.40 | 537.59 | 528.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 536.40 | 537.59 | 528.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 529.90 | 535.09 | 531.34 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 526.80 | 529.71 | 529.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 524.60 | 528.69 | 529.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 521.90 | 521.58 | 524.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 523.50 | 521.58 | 524.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 524.40 | 522.14 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 525.25 | 522.14 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 524.00 | 522.51 | 524.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 524.70 | 522.51 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 524.25 | 522.86 | 524.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:30:00 | 524.95 | 522.86 | 524.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 524.00 | 523.09 | 524.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 523.30 | 523.24 | 524.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 523.70 | 524.80 | 524.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 522.80 | 524.40 | 524.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 519.80 | 520.82 | 522.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 520.00 | 520.28 | 521.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 535.10 | 524.68 | 523.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 542.40 | 546.53 | 540.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:00:00 | 542.40 | 546.53 | 540.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 536.35 | 544.49 | 540.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 536.35 | 544.49 | 540.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 533.25 | 542.24 | 539.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 533.20 | 542.24 | 539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 533.80 | 538.32 | 538.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 533.20 | 538.32 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 533.55 | 537.36 | 537.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 532.15 | 535.78 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 530.80 | 534.45 | 535.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 531.05 | 533.07 | 534.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 539.25 | 533.44 | 531.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 549.00 | 549.33 | 543.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 551.00 | 549.33 | 543.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 602.75 | 589.64 | 574.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 617.65 | 596.41 | 585.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 612.55 | 624.54 | 626.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 579.20 | 559.86 | 568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 580.00 | 563.89 | 569.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 572.55 | 565.62 | 570.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:45:00 | 573.70 | 569.81 | 570.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 582.80 | 576.00 | 573.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 576.20 | 581.19 | 578.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 575.05 | 579.97 | 577.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 572.80 | 579.97 | 577.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 570.80 | 578.13 | 577.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 570.80 | 578.13 | 577.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 566.50 | 575.81 | 576.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 562.25 | 573.09 | 575.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 538.40 | 536.82 | 541.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 549.00 | 539.26 | 542.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 548.50 | 539.26 | 542.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 546.30 | 540.67 | 542.82 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 561.80 | 544.89 | 544.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 570.55 | 555.54 | 550.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 536.45 | 553.29 | 554.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 533.15 | 549.27 | 552.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 547.60 | 545.99 | 549.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 547.60 | 545.99 | 549.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 541.85 | 545.50 | 548.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 531.15 | 545.50 | 548.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:30:00 | 533.40 | 525.45 | 531.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 538.55 | 531.91 | 532.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 548.65 | 538.86 | 536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 541.70 | 541.73 | 538.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 541.70 | 541.73 | 538.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 534.75 | 540.22 | 538.76 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 535.15 | 537.76 | 538.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 525.70 | 535.35 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 533.65 | 533.56 | 535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 553.45 | 537.45 | 536.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 561.80 | 551.12 | 545.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 539.15 | 552.99 | 549.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 539.60 | 550.31 | 548.31 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 533.00 | 544.32 | 545.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 526.50 | 537.22 | 541.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 524.80 | 532.20 | 535.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 525.10 | 529.77 | 531.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 558.20 | 540.61 | 536.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 548.85 | 549.00 | 543.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 548.85 | 549.00 | 543.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 543.80 | 547.99 | 544.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.60 | 547.99 | 544.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 553.90 | 549.17 | 545.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 569.35 | 553.93 | 551.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 553.80 | 568.90 | 572.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 544.50 | 543.99 | 551.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 544.50 | 543.99 | 551.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 532.10 | 521.25 | 524.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 535.50 | 521.25 | 524.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 533.10 | 523.62 | 525.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 533.10 | 523.62 | 525.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 525.40 | 526.07 | 526.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 516.05 | 526.07 | 526.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 527.50 | 524.95 | 525.39 | SL hit (close>static) qty=1.00 sl=526.50 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 511.10 | 502.19 | 501.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 512.45 | 505.55 | 502.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 497.00 | 505.58 | 503.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 497.35 | 503.93 | 503.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 501.00 | 503.35 | 503.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 491.00 | 499.74 | 501.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 503.00 | 497.54 | 499.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 501.75 | 498.39 | 499.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 502.00 | 498.39 | 499.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 499.95 | 499.80 | 500.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 497.50 | 499.49 | 500.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 472.62 | 490.36 | 495.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 486.10 | 482.87 | 488.70 | SL hit (close>ema200) qty=0.50 sl=482.87 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 499.80 | 490.15 | 489.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 506.25 | 497.21 | 493.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 542.85 | 547.76 | 530.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:00:00 | 542.85 | 547.76 | 530.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 538.25 | 544.57 | 535.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 571.10 | 544.57 | 535.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 545.55 | 553.08 | 549.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 545.00 | 552.09 | 548.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 543.30 | 547.07 | 548.55 | Break + close below crossover candle low |

### Cycle 210 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 560.10 | 549.68 | 549.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 566.75 | 553.09 | 551.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 561.95 | 565.57 | 560.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 561.95 | 565.57 | 560.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 559.45 | 564.34 | 560.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 559.60 | 564.34 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 561.00 | 563.68 | 560.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 559.15 | 563.68 | 560.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 559.00 | 562.74 | 560.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 564.30 | 563.12 | 560.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 564.95 | 560.58 | 560.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 566.30 | 561.72 | 560.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 625.10 | 641.00 | 627.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 631.55 | 639.11 | 627.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 648.55 | 636.79 | 627.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 676.00 | 659.80 | 658.48 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 656.05 | 661.14 | 661.39 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 664.10 | 661.47 | 661.26 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 658.75 | 660.98 | 661.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 588.55 | 646.50 | 654.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 605.70 | 603.01 | 621.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 605.70 | 603.01 | 621.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 617.45 | 606.95 | 618.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 617.45 | 606.95 | 618.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 614.75 | 608.51 | 618.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 609.75 | 610.69 | 617.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-26 10:15:00 | 230.19 | 2023-05-31 13:15:00 | 231.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-05-26 12:15:00 | 230.24 | 2023-05-31 13:15:00 | 231.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-05-31 10:00:00 | 230.15 | 2023-05-31 13:15:00 | 231.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-06-26 12:45:00 | 318.39 | 2023-06-26 13:15:00 | 326.84 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2023-07-13 09:15:00 | 317.27 | 2023-07-13 11:15:00 | 313.60 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-07-13 10:00:00 | 316.77 | 2023-07-13 11:15:00 | 313.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-07-13 11:15:00 | 316.00 | 2023-07-13 11:15:00 | 313.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-07-21 09:15:00 | 321.40 | 2023-07-31 09:15:00 | 353.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-28 15:15:00 | 350.40 | 2023-08-29 09:15:00 | 352.82 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-09-29 10:45:00 | 343.02 | 2023-09-29 12:15:00 | 346.42 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-09-29 12:00:00 | 343.00 | 2023-09-29 12:15:00 | 346.42 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-10-04 10:15:00 | 356.21 | 2023-10-09 10:15:00 | 351.19 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-10-05 15:15:00 | 354.00 | 2023-10-09 10:15:00 | 351.19 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-10-06 10:00:00 | 353.98 | 2023-10-09 10:15:00 | 351.19 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-10-06 11:00:00 | 354.44 | 2023-10-09 10:15:00 | 351.19 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-10-11 09:15:00 | 357.50 | 2023-10-11 14:15:00 | 352.73 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-10-11 10:30:00 | 355.21 | 2023-10-11 14:15:00 | 352.73 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-10-11 13:00:00 | 356.00 | 2023-10-11 14:15:00 | 352.73 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-10-20 11:15:00 | 348.52 | 2023-10-23 14:15:00 | 331.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:15:00 | 348.52 | 2023-10-25 09:15:00 | 313.67 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-06 09:15:00 | 333.35 | 2023-11-07 09:15:00 | 326.92 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-11-06 12:45:00 | 330.80 | 2023-11-07 09:15:00 | 326.92 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-11-06 15:00:00 | 331.00 | 2023-11-07 09:15:00 | 326.92 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-11-15 10:45:00 | 313.20 | 2023-11-17 09:15:00 | 325.18 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2023-11-16 09:30:00 | 314.53 | 2023-11-17 09:15:00 | 325.18 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2023-11-16 10:30:00 | 315.34 | 2023-11-17 09:15:00 | 325.18 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2023-11-23 14:45:00 | 316.70 | 2023-11-29 09:15:00 | 326.80 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2023-11-23 15:15:00 | 316.65 | 2023-11-29 09:15:00 | 326.80 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2023-11-24 10:15:00 | 316.01 | 2023-11-29 09:15:00 | 326.80 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2023-12-07 10:30:00 | 348.16 | 2023-12-08 12:15:00 | 343.53 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-12-07 12:15:00 | 347.00 | 2023-12-08 12:15:00 | 343.53 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-12-07 15:15:00 | 346.76 | 2023-12-08 12:15:00 | 343.53 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-19 09:15:00 | 364.11 | 2023-12-20 13:15:00 | 349.08 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2023-12-19 10:45:00 | 366.18 | 2023-12-20 13:15:00 | 349.08 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2024-01-02 12:15:00 | 378.24 | 2024-01-02 13:15:00 | 377.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-01-10 15:15:00 | 368.05 | 2024-01-11 10:15:00 | 371.39 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-01-30 14:15:00 | 345.65 | 2024-01-31 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-01-30 15:00:00 | 346.16 | 2024-01-31 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-01-31 09:45:00 | 346.00 | 2024-01-31 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-02-01 15:15:00 | 359.19 | 2024-02-06 12:15:00 | 395.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-02 13:45:00 | 357.65 | 2024-02-06 12:15:00 | 393.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-05 09:15:00 | 360.57 | 2024-02-07 09:15:00 | 396.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-16 12:15:00 | 340.05 | 2024-02-16 12:15:00 | 340.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-03-01 13:15:00 | 328.06 | 2024-03-02 12:15:00 | 331.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-03-01 14:15:00 | 328.21 | 2024-03-02 12:15:00 | 331.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-03-05 14:15:00 | 340.74 | 2024-03-06 09:15:00 | 332.31 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-03-14 13:00:00 | 363.33 | 2024-03-19 14:15:00 | 362.06 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-03-15 10:30:00 | 361.38 | 2024-03-19 14:15:00 | 362.06 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-03-26 11:30:00 | 383.00 | 2024-03-27 11:15:00 | 376.06 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-03-26 15:15:00 | 379.00 | 2024-03-27 11:15:00 | 376.06 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-04-18 11:15:00 | 498.98 | 2024-04-24 14:15:00 | 485.05 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-04-19 13:15:00 | 495.99 | 2024-04-24 14:15:00 | 485.05 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-04-22 09:15:00 | 499.28 | 2024-04-24 14:15:00 | 485.05 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-04-22 10:30:00 | 494.56 | 2024-04-24 14:15:00 | 485.05 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-05-08 09:45:00 | 485.97 | 2024-05-09 15:15:00 | 476.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-05-09 09:15:00 | 485.08 | 2024-05-09 15:15:00 | 476.60 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-05-17 11:15:00 | 477.19 | 2024-05-17 13:15:00 | 483.78 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-28 11:15:00 | 458.23 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-05-28 11:45:00 | 458.21 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-05-29 10:00:00 | 458.63 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-05-31 15:15:00 | 458.00 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-06-13 13:00:00 | 443.20 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-13 14:00:00 | 443.18 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-14 09:15:00 | 444.40 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2024-06-21 09:15:00 | 466.47 | 2024-06-24 09:15:00 | 453.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2024-06-21 12:15:00 | 472.13 | 2024-06-24 09:15:00 | 453.00 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-07-03 09:15:00 | 444.70 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-03 10:30:00 | 445.25 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-07-08 13:30:00 | 446.57 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-23 12:15:00 | 404.20 | 2024-07-24 10:15:00 | 422.84 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-07-29 09:15:00 | 430.91 | 2024-08-02 11:15:00 | 436.18 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-07-30 09:15:00 | 429.03 | 2024-08-02 11:15:00 | 436.18 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-08-06 14:15:00 | 404.87 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-08-07 14:15:00 | 409.80 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-08-07 15:00:00 | 409.94 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-08-08 15:00:00 | 409.60 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-08-09 10:15:00 | 410.60 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-09 11:00:00 | 411.00 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-08-20 10:30:00 | 404.00 | 2024-08-21 13:15:00 | 406.32 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-08-20 14:45:00 | 404.10 | 2024-08-21 13:15:00 | 406.32 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-09-04 12:30:00 | 395.60 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-05 12:00:00 | 395.78 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-09-05 12:30:00 | 395.00 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-05 13:30:00 | 395.77 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-20 10:30:00 | 456.80 | 2024-09-25 09:15:00 | 502.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 12:30:00 | 454.73 | 2024-09-25 09:15:00 | 500.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 14:00:00 | 454.14 | 2024-09-25 09:15:00 | 499.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 15:00:00 | 455.00 | 2024-09-25 09:15:00 | 500.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-23 13:45:00 | 459.94 | 2024-09-25 09:15:00 | 503.80 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2024-09-24 09:15:00 | 458.00 | 2024-09-25 09:15:00 | 502.70 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2024-09-24 10:15:00 | 457.00 | 2024-09-25 09:15:00 | 503.14 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2024-09-24 11:45:00 | 457.40 | 2024-09-25 10:15:00 | 505.93 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2024-10-15 10:30:00 | 515.94 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2024-10-15 15:00:00 | 502.44 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-16 09:15:00 | 502.89 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-17 09:15:00 | 502.44 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-17 14:00:00 | 512.00 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-10-28 15:15:00 | 423.00 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-10-29 10:30:00 | 421.90 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-29 11:30:00 | 422.05 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-29 12:15:00 | 422.85 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-10-31 14:45:00 | 433.15 | 2024-11-04 09:15:00 | 426.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-11-05 09:45:00 | 433.15 | 2024-11-08 12:15:00 | 439.85 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2024-11-05 12:30:00 | 432.50 | 2024-11-08 12:15:00 | 439.85 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2024-11-18 13:30:00 | 414.15 | 2024-11-18 14:15:00 | 418.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-11-22 14:30:00 | 407.90 | 2024-11-25 09:15:00 | 417.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-12-03 10:45:00 | 484.95 | 2024-12-04 09:15:00 | 533.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-15 13:30:00 | 428.40 | 2025-01-22 11:15:00 | 406.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:30:00 | 427.95 | 2025-01-22 11:15:00 | 407.36 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-01-17 10:45:00 | 428.00 | 2025-01-22 12:15:00 | 406.55 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-01-17 15:15:00 | 428.80 | 2025-01-22 12:15:00 | 406.60 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-01-15 13:30:00 | 428.40 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-01-17 09:30:00 | 427.95 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-01-17 10:45:00 | 428.00 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-01-17 15:15:00 | 428.80 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-01-21 10:15:00 | 422.45 | 2025-01-24 13:15:00 | 403.51 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-01-21 15:15:00 | 424.75 | 2025-01-27 09:15:00 | 401.33 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2025-01-21 10:15:00 | 422.45 | 2025-01-27 12:15:00 | 409.00 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-01-21 15:15:00 | 424.75 | 2025-01-27 12:15:00 | 409.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-02-11 13:45:00 | 350.00 | 2025-02-14 13:15:00 | 333.50 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-02-11 13:45:00 | 350.00 | 2025-02-14 14:15:00 | 339.20 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-02-11 15:00:00 | 351.05 | 2025-02-17 09:15:00 | 332.50 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2025-02-11 15:00:00 | 351.05 | 2025-02-17 12:15:00 | 338.50 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-02-12 09:15:00 | 340.95 | 2025-02-19 09:15:00 | 356.10 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-02-13 11:00:00 | 347.55 | 2025-02-19 09:15:00 | 356.10 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-02-24 11:15:00 | 359.65 | 2025-02-25 11:15:00 | 349.95 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest1 | 2025-02-28 09:15:00 | 340.60 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2025-02-28 10:45:00 | 345.40 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest1 | 2025-02-28 12:30:00 | 345.40 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-25 13:15:00 | 497.05 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-03-26 09:45:00 | 502.75 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2025-04-02 12:30:00 | 510.95 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest2 | 2025-04-03 10:00:00 | 498.25 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-04-08 10:30:00 | 456.70 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-04-08 15:15:00 | 456.50 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-04-11 09:30:00 | 455.75 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-04-22 09:15:00 | 485.70 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-04-22 10:15:00 | 484.30 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-04-22 14:00:00 | 484.65 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-06 11:30:00 | 454.80 | 2025-05-08 15:15:00 | 432.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:45:00 | 454.95 | 2025-05-08 15:15:00 | 432.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 454.80 | 2025-05-12 09:15:00 | 455.65 | STOP_HIT | 0.50 | -0.19% |
| SELL | retest2 | 2025-05-06 12:45:00 | 454.95 | 2025-05-12 09:15:00 | 455.65 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-06-17 12:00:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 12:30:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 13:45:00 | 500.10 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-17 15:15:00 | 498.00 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-07-14 13:30:00 | 527.25 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-07-14 14:15:00 | 527.40 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-08-08 09:15:00 | 516.15 | 2025-08-18 13:15:00 | 514.90 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-08-26 15:15:00 | 476.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-09-01 09:15:00 | 472.30 | STOP_HIT | 0.50 | 5.75% |
| BUY | retest2 | 2025-09-03 09:15:00 | 484.40 | 2025-09-22 15:15:00 | 517.45 | STOP_HIT | 1.00 | 6.82% |
| BUY | retest2 | 2025-09-25 09:45:00 | 527.70 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-25 10:30:00 | 527.30 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-25 11:45:00 | 526.65 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-09-25 12:15:00 | 526.45 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-29 11:30:00 | 508.50 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-29 14:45:00 | 510.00 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-01 10:45:00 | 509.85 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-09 09:15:00 | 519.45 | 2025-10-10 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-23 10:15:00 | 511.65 | 2025-10-23 13:15:00 | 517.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-17 09:15:00 | 508.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-18 11:15:00 | 517.70 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-12-03 14:15:00 | 523.30 | 2025-12-04 09:15:00 | 531.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-08 10:30:00 | 519.80 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-08 15:15:00 | 520.00 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-12-16 12:00:00 | 530.80 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-17 09:30:00 | 531.05 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-31 09:15:00 | 617.65 | 2026-01-07 15:15:00 | 616.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-14 12:00:00 | 572.55 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-16 10:45:00 | 573.70 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-01 12:15:00 | 531.15 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-03 09:30:00 | 533.40 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-02-03 12:30:00 | 538.55 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-13 15:15:00 | 524.80 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-02-17 14:30:00 | 525.10 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-02-24 09:15:00 | 569.35 | 2026-03-02 11:15:00 | 563.45 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-03-12 09:15:00 | 516.05 | 2026-03-12 12:15:00 | 527.50 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-13 15:15:00 | 495.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-16 14:15:00 | 497.50 | STOP_HIT | 0.50 | 4.56% |
| BUY | retest2 | 2026-03-19 12:00:00 | 501.00 | 2026-03-19 12:15:00 | 496.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-23 10:15:00 | 472.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-24 09:15:00 | 486.10 | STOP_HIT | 0.50 | 2.29% |
| BUY | retest2 | 2026-04-01 09:15:00 | 571.10 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2026-04-02 11:45:00 | 545.55 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-02 12:30:00 | 545.00 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-10 09:30:00 | 564.30 | 2026-04-13 09:15:00 | 553.05 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-04-21 09:15:00 | 648.55 | 2026-04-24 13:15:00 | 648.20 | STOP_HIT | 1.00 | -0.05% |
