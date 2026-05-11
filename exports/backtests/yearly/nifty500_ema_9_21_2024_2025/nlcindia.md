# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 328.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 115 |
| ALERT2 | 113 |
| ALERT2_SKIP | 47 |
| ALERT3 | 296 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 115 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 119 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 104
- **Target hits / Stop hits / Partials:** 2 / 119 / 6
- **Avg / median % per leg:** -0.68% / -1.09%
- **Sum % (uncompounded):** -85.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 7 | 11.7% | 1 | 58 | 1 | -1.06% | -63.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.26% | 1.3% |
| BUY @ 3rd Alert (retest2) | 55 | 5 | 9.1% | 1 | 54 | 0 | -1.18% | -64.7% |
| SELL (all) | 67 | 16 | 23.9% | 1 | 61 | 5 | -0.34% | -22.6% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.76% | 2.3% |
| SELL @ 3rd Alert (retest2) | 64 | 14 | 21.9% | 1 | 58 | 5 | -0.39% | -24.9% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 7 | 1 | 0.45% | 3.6% |
| retest2 (combined) | 119 | 19 | 16.0% | 2 | 112 | 5 | -0.75% | -89.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 223.20 | 218.64 | 218.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 223.95 | 221.36 | 219.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 222.10 | 226.59 | 224.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 222.10 | 226.59 | 224.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 222.10 | 226.59 | 224.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 11:45:00 | 227.00 | 226.23 | 224.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:45:00 | 227.10 | 226.05 | 224.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:45:00 | 229.25 | 226.80 | 225.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 237.90 | 240.20 | 240.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 237.90 | 240.20 | 240.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 237.00 | 239.56 | 240.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 220.30 | 219.22 | 221.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:00:00 | 220.30 | 219.22 | 221.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 222.55 | 220.06 | 221.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 222.55 | 220.06 | 221.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 234.40 | 222.93 | 222.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 237.10 | 225.76 | 224.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 232.25 | 238.06 | 233.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 232.25 | 238.06 | 233.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 232.25 | 238.06 | 233.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 228.65 | 238.06 | 233.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 208.65 | 232.18 | 231.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 208.65 | 232.18 | 231.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 201.65 | 226.08 | 228.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 198.60 | 210.65 | 219.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 217.60 | 206.96 | 212.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 217.60 | 206.96 | 212.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 217.60 | 206.96 | 212.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 217.60 | 206.96 | 212.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 219.90 | 209.55 | 212.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 216.50 | 210.65 | 213.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 216.95 | 212.59 | 213.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 217.80 | 214.78 | 214.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 217.80 | 214.78 | 214.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 220.95 | 217.37 | 215.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 234.85 | 234.87 | 232.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 234.34 | 234.87 | 232.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 232.02 | 233.86 | 232.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 233.45 | 232.90 | 232.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:30:00 | 233.05 | 233.22 | 232.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 233.11 | 233.20 | 232.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 233.16 | 233.12 | 232.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 232.46 | 232.98 | 232.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 232.46 | 232.98 | 232.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 232.35 | 232.86 | 232.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 232.35 | 232.86 | 232.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 231.55 | 232.60 | 232.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 229.60 | 232.00 | 232.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 229.60 | 232.00 | 232.14 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 233.46 | 232.29 | 232.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 12:15:00 | 235.66 | 233.10 | 232.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 234.00 | 234.22 | 233.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 234.00 | 234.22 | 233.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 234.00 | 234.22 | 233.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 230.44 | 234.22 | 233.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 234.02 | 234.18 | 233.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 234.67 | 234.18 | 233.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 234.00 | 234.15 | 233.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:45:00 | 236.19 | 234.57 | 233.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 231.52 | 234.42 | 234.37 | SL hit (close<static) qty=1.00 sl=232.80 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 15:15:00 | 231.20 | 233.78 | 234.08 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 236.33 | 234.31 | 234.10 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 232.25 | 234.17 | 234.29 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 236.51 | 234.58 | 234.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 244.20 | 236.50 | 235.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 14:15:00 | 243.10 | 244.14 | 241.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 243.10 | 244.14 | 241.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 243.10 | 244.14 | 241.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 243.10 | 244.14 | 241.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 239.74 | 243.50 | 241.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 238.99 | 243.50 | 241.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 240.49 | 242.90 | 241.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:15:00 | 239.30 | 242.90 | 241.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 239.70 | 241.64 | 241.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 239.70 | 241.64 | 241.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 240.40 | 241.39 | 241.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 237.67 | 241.39 | 241.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 237.91 | 240.70 | 240.85 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 242.07 | 240.99 | 240.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 244.00 | 241.60 | 241.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 256.55 | 258.08 | 254.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 256.55 | 258.08 | 254.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 256.55 | 258.08 | 254.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 255.85 | 258.08 | 254.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 266.09 | 268.01 | 265.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:15:00 | 263.31 | 268.01 | 265.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 263.23 | 267.05 | 264.96 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 255.21 | 262.99 | 263.71 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 270.71 | 265.12 | 264.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 282.00 | 272.38 | 268.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 281.90 | 282.22 | 277.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 12:00:00 | 281.90 | 282.22 | 277.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 277.50 | 280.81 | 278.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 276.24 | 280.81 | 278.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 278.48 | 280.34 | 278.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 274.36 | 280.34 | 278.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 294.50 | 283.17 | 279.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:00:00 | 308.97 | 294.82 | 287.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 283.50 | 288.31 | 288.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 283.50 | 288.31 | 288.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 275.99 | 283.19 | 285.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 278.79 | 278.78 | 282.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 11:30:00 | 275.90 | 278.78 | 282.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 277.80 | 268.64 | 270.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 277.80 | 268.64 | 270.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 284.06 | 271.73 | 272.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 284.06 | 271.73 | 272.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 283.60 | 274.10 | 273.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 288.72 | 283.06 | 280.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 287.00 | 287.26 | 283.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 290.27 | 287.26 | 283.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:45:00 | 289.90 | 288.09 | 284.82 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 285.28 | 287.40 | 285.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-29 13:15:00 | 285.28 | 287.40 | 285.32 | SL hit (close<ema400) qty=1.00 sl=285.32 alert=retest1 |

### Cycle 18 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 286.05 | 289.96 | 290.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 284.00 | 287.63 | 289.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 266.05 | 261.11 | 266.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 11:15:00 | 266.05 | 261.11 | 266.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 266.05 | 261.11 | 266.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 266.05 | 261.11 | 266.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 268.70 | 262.63 | 266.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 268.15 | 262.63 | 266.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 269.20 | 263.94 | 266.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 268.30 | 263.94 | 266.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 266.50 | 264.89 | 266.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 264.20 | 264.89 | 266.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 262.95 | 264.50 | 266.43 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 267.35 | 266.10 | 266.08 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 264.90 | 265.97 | 266.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 261.35 | 265.05 | 265.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 11:15:00 | 265.15 | 264.92 | 265.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 11:15:00 | 265.15 | 264.92 | 265.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 265.15 | 264.92 | 265.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:30:00 | 266.50 | 264.92 | 265.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 266.70 | 265.28 | 265.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:00:00 | 266.70 | 265.28 | 265.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 267.25 | 265.67 | 265.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 267.25 | 265.67 | 265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 267.80 | 266.10 | 265.91 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 263.30 | 265.80 | 265.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 259.55 | 264.55 | 265.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 262.35 | 262.13 | 263.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 262.35 | 262.13 | 263.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 259.70 | 261.64 | 263.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 260.00 | 261.64 | 263.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 262.75 | 261.26 | 262.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 262.55 | 261.26 | 262.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 263.40 | 261.69 | 262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 263.75 | 261.69 | 262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 263.45 | 262.04 | 262.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 261.45 | 262.32 | 262.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 261.10 | 262.21 | 262.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 264.20 | 262.60 | 262.84 | SL hit (close>static) qty=1.00 sl=263.90 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 271.35 | 263.77 | 263.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 273.60 | 270.25 | 267.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 271.90 | 272.67 | 270.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:45:00 | 272.00 | 272.67 | 270.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 270.90 | 272.31 | 270.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 270.75 | 272.31 | 270.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 272.00 | 272.25 | 271.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 272.00 | 272.25 | 271.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 275.10 | 274.64 | 273.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 274.70 | 274.64 | 273.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 273.10 | 274.56 | 273.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 273.10 | 274.56 | 273.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 273.65 | 274.38 | 273.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 281.00 | 274.38 | 273.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 274.65 | 277.21 | 277.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 274.65 | 277.21 | 277.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 13:15:00 | 273.00 | 275.43 | 276.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 10:15:00 | 275.35 | 274.29 | 275.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 10:15:00 | 275.35 | 274.29 | 275.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 275.35 | 274.29 | 275.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 275.35 | 274.29 | 275.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 273.75 | 274.18 | 275.33 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 276.20 | 275.64 | 275.61 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 275.45 | 275.60 | 275.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 12:15:00 | 274.05 | 275.29 | 275.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 274.80 | 274.51 | 274.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 274.80 | 274.51 | 274.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 274.80 | 274.51 | 274.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 275.10 | 274.51 | 274.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 272.35 | 274.08 | 274.73 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 278.25 | 274.49 | 274.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 13:15:00 | 281.65 | 276.56 | 275.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 11:15:00 | 275.95 | 278.00 | 276.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 275.95 | 278.00 | 276.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 275.95 | 278.00 | 276.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:45:00 | 275.85 | 278.00 | 276.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 273.75 | 277.15 | 276.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 273.75 | 277.15 | 276.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 270.15 | 274.93 | 275.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 268.75 | 273.69 | 274.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 266.55 | 264.12 | 267.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 266.55 | 264.12 | 267.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 266.55 | 264.12 | 267.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 267.35 | 264.12 | 267.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 267.30 | 265.39 | 267.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:30:00 | 267.65 | 265.39 | 267.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 267.65 | 265.84 | 267.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 267.35 | 265.84 | 267.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 267.65 | 266.21 | 267.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:15:00 | 271.90 | 266.21 | 267.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 271.90 | 267.34 | 268.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 272.25 | 267.34 | 268.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 271.40 | 268.16 | 268.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:45:00 | 271.75 | 268.16 | 268.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 273.00 | 269.12 | 268.73 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 266.00 | 268.38 | 268.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 10:15:00 | 261.30 | 266.96 | 268.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 272.95 | 266.70 | 267.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 272.95 | 266.70 | 267.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 272.95 | 266.70 | 267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 272.95 | 266.70 | 267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 272.30 | 267.82 | 267.58 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 269.05 | 270.63 | 270.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 268.45 | 270.20 | 270.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 266.10 | 264.89 | 266.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:00:00 | 266.10 | 264.89 | 266.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 266.80 | 265.27 | 266.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 266.95 | 265.27 | 266.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 263.75 | 264.97 | 266.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 266.85 | 264.97 | 266.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 272.15 | 265.91 | 266.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 272.15 | 265.91 | 266.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 272.50 | 267.23 | 267.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 274.00 | 270.12 | 268.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 282.35 | 282.71 | 278.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:45:00 | 282.00 | 282.71 | 278.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 281.20 | 282.88 | 280.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 280.30 | 282.88 | 280.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 281.10 | 282.52 | 280.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 279.50 | 282.52 | 280.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 277.35 | 281.49 | 279.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 277.35 | 281.49 | 279.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 276.30 | 280.45 | 279.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 276.30 | 280.45 | 279.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 14:15:00 | 274.70 | 278.91 | 279.02 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 281.20 | 279.18 | 279.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 291.25 | 282.16 | 280.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 12:15:00 | 285.10 | 285.13 | 282.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:00:00 | 285.10 | 285.13 | 282.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 281.80 | 284.25 | 282.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 281.80 | 284.25 | 282.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 282.95 | 283.99 | 282.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 283.90 | 283.99 | 282.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 284.25 | 284.07 | 283.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 15:15:00 | 282.75 | 284.44 | 284.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 282.75 | 284.44 | 284.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 280.50 | 283.65 | 284.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 270.90 | 267.79 | 272.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 268.50 | 267.79 | 272.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 269.90 | 268.35 | 271.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 269.90 | 268.35 | 271.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 274.35 | 269.55 | 271.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 274.35 | 269.55 | 271.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 275.50 | 270.74 | 271.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 275.25 | 270.74 | 271.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 278.40 | 273.11 | 272.90 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 269.95 | 272.88 | 273.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 15:15:00 | 269.50 | 272.20 | 272.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 267.60 | 267.55 | 269.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 14:00:00 | 267.60 | 267.55 | 269.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 267.50 | 267.51 | 268.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 269.40 | 267.51 | 268.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 267.75 | 267.55 | 268.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 269.00 | 267.55 | 268.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 268.05 | 267.30 | 268.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 268.75 | 267.30 | 268.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 269.40 | 267.72 | 268.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 269.40 | 267.72 | 268.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 269.00 | 267.98 | 268.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 267.75 | 267.98 | 268.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:45:00 | 268.25 | 268.08 | 268.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 270.30 | 268.53 | 268.60 | SL hit (close>static) qty=1.00 sl=270.20 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 11:15:00 | 270.10 | 268.84 | 268.74 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 267.15 | 268.50 | 268.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 263.55 | 266.93 | 267.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 262.75 | 262.17 | 264.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 11:45:00 | 262.90 | 262.17 | 264.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 264.35 | 262.77 | 264.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 264.35 | 262.77 | 264.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 264.95 | 263.21 | 264.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 264.85 | 263.21 | 264.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 264.90 | 263.55 | 264.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 262.00 | 263.55 | 264.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 248.90 | 257.11 | 260.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 235.80 | 247.63 | 253.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 247.20 | 244.51 | 244.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 249.95 | 246.67 | 245.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 250.50 | 250.86 | 248.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 250.50 | 250.86 | 248.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 252.95 | 251.52 | 249.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 256.45 | 251.52 | 249.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 254.80 | 254.36 | 251.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 255.75 | 253.84 | 253.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 253.75 | 257.70 | 257.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 253.75 | 257.70 | 257.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 252.20 | 256.60 | 257.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 10:15:00 | 235.00 | 234.59 | 237.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 10:45:00 | 234.90 | 234.59 | 237.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 238.15 | 235.30 | 237.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 238.15 | 235.30 | 237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 236.10 | 235.46 | 237.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 235.15 | 235.72 | 237.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 09:15:00 | 241.25 | 237.44 | 237.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 09:15:00 | 241.25 | 237.44 | 237.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 14:15:00 | 251.55 | 243.86 | 240.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 260.65 | 260.92 | 255.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 14:30:00 | 261.00 | 260.92 | 255.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 259.95 | 260.48 | 259.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 259.95 | 260.48 | 259.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 259.05 | 260.20 | 259.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 259.05 | 260.20 | 259.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 258.15 | 259.79 | 259.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:15:00 | 259.00 | 259.79 | 259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 256.60 | 259.02 | 259.06 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 262.15 | 259.06 | 258.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 263.65 | 259.98 | 259.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 263.60 | 263.65 | 262.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:15:00 | 265.80 | 263.65 | 262.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 264.70 | 264.40 | 262.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 264.70 | 264.40 | 262.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 264.00 | 264.96 | 263.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 264.00 | 264.96 | 263.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 264.00 | 264.77 | 263.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 264.55 | 264.77 | 263.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 262.85 | 264.38 | 263.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 262.85 | 264.38 | 263.55 | SL hit (close<ema400) qty=1.00 sl=263.55 alert=retest1 |

### Cycle 46 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 263.30 | 266.36 | 266.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 10:15:00 | 262.90 | 264.56 | 265.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 257.85 | 257.58 | 259.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 257.85 | 257.58 | 259.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 259.00 | 257.77 | 259.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:45:00 | 255.90 | 257.82 | 258.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:45:00 | 255.90 | 257.39 | 258.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 243.10 | 250.34 | 253.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 253.45 | 250.34 | 253.10 | SL hit (close>static) qty=0.50 sl=250.34 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 242.05 | 240.81 | 240.71 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 239.65 | 240.57 | 240.61 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 241.60 | 240.78 | 240.70 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 238.75 | 240.37 | 240.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 238.00 | 239.59 | 240.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 240.10 | 239.69 | 240.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 11:15:00 | 240.10 | 239.69 | 240.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 240.10 | 239.69 | 240.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:45:00 | 240.00 | 239.69 | 240.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 240.30 | 239.81 | 240.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:45:00 | 240.00 | 239.81 | 240.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 237.70 | 239.39 | 239.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 233.75 | 238.26 | 239.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 241.00 | 238.63 | 238.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 241.00 | 238.63 | 238.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 242.90 | 240.70 | 239.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 240.85 | 240.90 | 240.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:30:00 | 241.10 | 240.90 | 240.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 240.95 | 240.91 | 240.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 240.45 | 240.91 | 240.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 243.10 | 242.71 | 241.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 243.10 | 242.71 | 241.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 238.90 | 242.03 | 241.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 239.34 | 242.03 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 234.41 | 240.50 | 240.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 233.90 | 239.18 | 240.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 233.93 | 233.30 | 235.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:00:00 | 233.93 | 233.30 | 235.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 235.44 | 234.16 | 235.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 235.44 | 234.16 | 235.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 234.60 | 234.25 | 235.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 232.47 | 233.89 | 234.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 238.67 | 234.24 | 234.51 | SL hit (close>static) qty=1.00 sl=236.78 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 242.11 | 235.82 | 235.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 14:15:00 | 244.09 | 237.47 | 236.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 239.48 | 239.68 | 238.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 13:30:00 | 239.27 | 239.68 | 238.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 235.49 | 238.95 | 238.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 235.49 | 238.95 | 238.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 233.17 | 237.79 | 237.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 233.17 | 237.79 | 237.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 232.25 | 236.68 | 237.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 228.51 | 235.05 | 236.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 231.40 | 230.51 | 233.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:45:00 | 231.25 | 230.51 | 233.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 232.23 | 230.86 | 232.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:15:00 | 228.63 | 231.45 | 232.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 229.41 | 230.75 | 231.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 228.85 | 229.43 | 230.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 228.31 | 229.25 | 230.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 230.31 | 229.38 | 230.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 230.63 | 229.38 | 230.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 231.04 | 229.71 | 230.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 232.00 | 229.71 | 230.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 229.95 | 229.76 | 230.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 230.91 | 229.76 | 230.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 230.65 | 229.94 | 230.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 232.30 | 230.68 | 230.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 232.30 | 230.68 | 230.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 235.00 | 232.08 | 231.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 11:15:00 | 229.22 | 233.26 | 232.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 11:15:00 | 229.22 | 233.26 | 232.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 229.22 | 233.26 | 232.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 229.22 | 233.26 | 232.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 228.18 | 232.25 | 232.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:45:00 | 227.37 | 232.25 | 232.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 228.58 | 231.51 | 231.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 225.71 | 228.63 | 229.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 207.98 | 206.97 | 213.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 207.98 | 206.97 | 213.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 210.15 | 208.25 | 210.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 210.15 | 208.25 | 210.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 210.30 | 208.66 | 210.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 214.82 | 208.66 | 210.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 216.18 | 210.16 | 211.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 217.38 | 210.16 | 211.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 216.65 | 211.46 | 211.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:15:00 | 215.93 | 211.46 | 211.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 215.50 | 212.27 | 211.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 219.45 | 214.52 | 213.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 217.21 | 220.25 | 217.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 217.21 | 220.25 | 217.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 217.21 | 220.25 | 217.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 217.21 | 220.25 | 217.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 218.59 | 219.92 | 217.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 219.50 | 219.49 | 217.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 210.18 | 217.63 | 217.35 | SL hit (close<static) qty=1.00 sl=216.69 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 209.37 | 215.98 | 216.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 207.63 | 214.31 | 215.81 | Break + close below crossover candle low |

### Cycle 59 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 236.16 | 216.64 | 215.94 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 219.90 | 222.35 | 222.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 218.22 | 221.52 | 222.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 202.90 | 202.65 | 206.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 202.61 | 202.65 | 206.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 203.98 | 202.57 | 205.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 204.89 | 202.57 | 205.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 203.90 | 202.84 | 204.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 203.90 | 202.84 | 204.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 203.64 | 203.00 | 204.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 204.05 | 203.00 | 204.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 197.70 | 193.37 | 196.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 197.70 | 193.37 | 196.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 205.25 | 195.75 | 197.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 205.25 | 195.75 | 197.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 210.15 | 198.63 | 198.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 215.50 | 207.08 | 202.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 225.40 | 226.36 | 221.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 09:15:00 | 230.40 | 226.36 | 221.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 222.97 | 225.59 | 223.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 219.36 | 225.59 | 223.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 220.87 | 224.64 | 223.26 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 219.01 | 222.00 | 222.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 218.31 | 220.74 | 221.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 208.59 | 207.74 | 211.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:30:00 | 208.49 | 207.74 | 211.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 207.24 | 207.08 | 209.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 210.25 | 207.08 | 209.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 201.39 | 205.61 | 208.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:00:00 | 200.00 | 204.49 | 207.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 13:15:00 | 211.82 | 207.42 | 207.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 211.82 | 207.42 | 207.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 227.99 | 211.53 | 209.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 219.60 | 221.20 | 216.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 225.71 | 222.24 | 217.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 12:15:00 | 237.00 | 226.44 | 220.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 15:15:00 | 227.35 | 227.88 | 223.05 | SL hit (close<ema200) qty=0.50 sl=227.88 alert=retest1 |

### Cycle 64 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 220.59 | 223.45 | 223.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 217.90 | 222.08 | 223.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 220.70 | 219.31 | 220.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 220.70 | 219.31 | 220.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 220.70 | 219.31 | 220.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 226.79 | 219.31 | 220.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 222.24 | 219.89 | 220.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 225.75 | 219.89 | 220.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 224.98 | 220.91 | 221.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 224.98 | 220.91 | 221.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 225.95 | 221.92 | 221.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 12:15:00 | 229.49 | 223.43 | 222.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 226.75 | 226.86 | 224.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 11:00:00 | 226.75 | 226.86 | 224.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 226.49 | 226.60 | 225.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:15:00 | 225.18 | 226.60 | 225.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 225.18 | 226.32 | 225.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 228.33 | 226.32 | 225.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 09:15:00 | 251.16 | 247.34 | 243.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 246.21 | 252.20 | 252.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 243.55 | 248.86 | 250.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 245.89 | 244.44 | 246.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 245.89 | 244.44 | 246.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 245.89 | 244.44 | 246.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 247.27 | 244.44 | 246.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 245.48 | 244.65 | 246.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 247.19 | 244.65 | 246.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 247.25 | 245.17 | 246.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 244.40 | 244.94 | 246.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 242.50 | 244.23 | 245.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 244.59 | 244.07 | 245.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 242.94 | 244.07 | 245.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 243.47 | 243.41 | 244.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 243.47 | 243.41 | 244.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 241.50 | 242.96 | 243.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 247.60 | 244.76 | 244.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 247.60 | 244.76 | 244.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 248.20 | 245.45 | 244.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 240.64 | 246.79 | 246.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 240.64 | 246.79 | 246.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 240.64 | 246.79 | 246.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 240.64 | 246.79 | 246.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 240.66 | 245.56 | 245.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 237.36 | 242.74 | 244.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 227.50 | 227.44 | 233.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 227.50 | 227.44 | 233.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 227.50 | 227.44 | 233.28 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 238.05 | 230.93 | 230.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 247.56 | 238.45 | 234.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 239.59 | 240.94 | 238.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 239.59 | 240.94 | 238.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 245.80 | 246.29 | 244.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 248.19 | 246.29 | 244.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 242.00 | 245.43 | 244.45 | SL hit (close<static) qty=1.00 sl=244.10 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 237.61 | 243.21 | 243.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 235.33 | 237.65 | 239.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 230.88 | 230.14 | 233.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 12:15:00 | 226.15 | 229.38 | 232.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 12:30:00 | 226.93 | 227.54 | 229.49 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 223.82 | 226.28 | 228.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 222.25 | 225.18 | 227.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 222.82 | 225.18 | 227.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 222.27 | 221.91 | 223.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 226.35 | 223.05 | 223.78 | SL hit (close>ema400) qty=1.00 sl=223.78 alert=retest1 |

### Cycle 71 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 226.73 | 224.55 | 224.38 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 222.22 | 223.93 | 224.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 220.25 | 223.20 | 223.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 220.12 | 218.91 | 220.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 220.12 | 218.91 | 220.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 220.12 | 218.91 | 220.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 220.12 | 218.91 | 220.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 220.55 | 219.24 | 220.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 229.39 | 219.24 | 220.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 231.68 | 221.72 | 221.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 234.07 | 229.22 | 226.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 228.55 | 229.82 | 227.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 228.55 | 229.82 | 227.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 235.85 | 238.46 | 236.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 235.85 | 238.46 | 236.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 236.75 | 238.12 | 236.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 254.21 | 237.90 | 236.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 237.80 | 239.39 | 239.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 237.80 | 239.39 | 239.54 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 240.94 | 239.50 | 239.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 242.06 | 240.33 | 239.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 240.36 | 240.52 | 240.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:45:00 | 240.27 | 240.52 | 240.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 240.48 | 240.50 | 240.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 240.37 | 240.50 | 240.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 239.15 | 240.23 | 240.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 240.52 | 240.23 | 240.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 239.56 | 240.09 | 240.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:15:00 | 239.10 | 240.09 | 240.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 240.75 | 240.23 | 240.08 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 239.14 | 239.93 | 239.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 14:15:00 | 238.45 | 239.63 | 239.84 | Break + close below crossover candle low |

### Cycle 77 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 242.36 | 240.04 | 239.98 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 239.80 | 241.01 | 241.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 13:15:00 | 239.12 | 240.31 | 240.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 235.33 | 235.32 | 237.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 236.91 | 235.91 | 237.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 236.91 | 235.91 | 237.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 236.91 | 235.91 | 237.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 238.03 | 236.34 | 237.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 238.75 | 236.34 | 237.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 237.94 | 236.66 | 237.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 237.94 | 236.66 | 237.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 238.39 | 237.71 | 237.65 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 237.13 | 237.56 | 237.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 236.13 | 237.24 | 237.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 238.95 | 236.77 | 236.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 238.95 | 236.77 | 236.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 238.95 | 236.77 | 236.98 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 240.00 | 237.42 | 237.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 242.65 | 238.47 | 237.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 240.86 | 241.18 | 239.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:00:00 | 240.86 | 241.18 | 239.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 241.44 | 241.31 | 240.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:45:00 | 240.23 | 241.31 | 240.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 239.67 | 241.68 | 240.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 240.00 | 241.68 | 240.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 241.40 | 241.62 | 241.02 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 239.00 | 240.49 | 240.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 235.52 | 239.50 | 240.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 231.93 | 231.40 | 233.28 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:45:00 | 230.16 | 231.02 | 232.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 229.57 | 229.63 | 231.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 227.15 | 229.03 | 230.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 225.29 | 223.56 | 224.69 | SL hit (close>ema400) qty=1.00 sl=224.69 alert=retest1 |

### Cycle 83 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 227.48 | 225.71 | 225.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 230.30 | 227.20 | 226.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 227.50 | 228.49 | 227.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 227.50 | 228.49 | 227.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 227.50 | 228.49 | 227.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 227.94 | 228.49 | 227.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 227.68 | 228.33 | 227.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 227.75 | 228.33 | 227.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 227.75 | 228.21 | 227.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 229.56 | 228.21 | 227.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:30:00 | 228.26 | 228.24 | 227.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 230.43 | 228.19 | 227.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 226.94 | 227.97 | 227.84 | SL hit (close<static) qty=1.00 sl=227.10 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 226.73 | 227.72 | 227.74 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 229.03 | 227.98 | 227.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 231.65 | 228.87 | 228.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 229.67 | 229.95 | 229.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:30:00 | 229.70 | 229.95 | 229.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 227.88 | 229.54 | 229.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 227.88 | 229.54 | 229.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 229.00 | 229.43 | 229.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 232.42 | 229.43 | 229.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 226.82 | 229.62 | 229.55 | SL hit (close<static) qty=1.00 sl=227.56 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 226.79 | 229.06 | 229.30 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 229.48 | 228.47 | 228.37 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 227.20 | 228.22 | 228.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 226.35 | 227.84 | 228.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 11:15:00 | 227.26 | 227.22 | 227.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 226.96 | 227.22 | 227.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 226.52 | 227.08 | 227.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 226.01 | 227.08 | 227.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 226.47 | 226.97 | 227.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 227.80 | 227.14 | 227.44 | SL hit (close>static) qty=1.00 sl=227.50 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 10:15:00 | 227.05 | 225.21 | 225.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 227.10 | 225.59 | 225.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 12:15:00 | 227.11 | 227.32 | 226.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 12:15:00 | 227.11 | 227.32 | 226.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 227.11 | 227.32 | 226.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 227.08 | 227.32 | 226.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 227.18 | 227.40 | 226.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 227.18 | 227.40 | 226.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 227.02 | 227.24 | 226.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 226.63 | 227.24 | 226.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 226.99 | 227.19 | 226.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 228.60 | 227.51 | 226.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 233.62 | 234.57 | 234.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 233.62 | 234.57 | 234.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 231.14 | 233.38 | 234.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 231.79 | 231.77 | 232.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:00:00 | 231.79 | 231.77 | 232.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 232.88 | 231.99 | 232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 232.81 | 231.99 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 240.77 | 233.75 | 233.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 242.29 | 235.46 | 234.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 240.68 | 244.11 | 242.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 240.68 | 244.11 | 242.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 240.68 | 244.11 | 242.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 240.68 | 244.11 | 242.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 240.15 | 243.32 | 242.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 239.54 | 243.32 | 242.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 239.54 | 241.53 | 241.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 237.50 | 240.40 | 241.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 238.12 | 235.96 | 237.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 238.12 | 235.96 | 237.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 238.12 | 235.96 | 237.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 238.12 | 235.96 | 237.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 239.75 | 236.72 | 237.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 239.75 | 236.72 | 237.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 238.90 | 237.16 | 237.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 239.38 | 237.16 | 237.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 242.50 | 238.79 | 238.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 244.10 | 241.21 | 239.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 242.00 | 242.49 | 241.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 242.00 | 242.49 | 241.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 239.60 | 241.88 | 241.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 238.85 | 241.88 | 241.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 238.46 | 241.19 | 240.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 238.46 | 241.19 | 240.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 237.44 | 240.44 | 240.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 236.65 | 239.20 | 240.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 238.65 | 237.66 | 238.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 238.65 | 237.66 | 238.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 238.65 | 237.66 | 238.99 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 243.93 | 239.96 | 239.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 244.74 | 240.92 | 240.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 11:15:00 | 242.48 | 242.56 | 241.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:00:00 | 242.48 | 242.56 | 241.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 242.00 | 242.45 | 241.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 242.00 | 242.45 | 241.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 239.98 | 241.96 | 241.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 239.98 | 241.96 | 241.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 239.62 | 241.49 | 241.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:15:00 | 239.00 | 241.49 | 241.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 239.00 | 240.99 | 240.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 241.00 | 240.99 | 240.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 238.37 | 240.47 | 240.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 238.37 | 240.47 | 240.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 234.72 | 239.32 | 240.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 233.60 | 232.09 | 234.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 233.60 | 232.09 | 234.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 232.91 | 232.51 | 234.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 230.84 | 232.37 | 233.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 230.53 | 231.96 | 233.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 238.52 | 234.20 | 233.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 238.52 | 234.20 | 233.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 240.02 | 236.29 | 234.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 238.45 | 240.49 | 238.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 238.45 | 240.49 | 238.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 238.45 | 240.49 | 238.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 238.45 | 240.49 | 238.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 238.03 | 240.00 | 238.67 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 237.11 | 238.00 | 238.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 235.49 | 237.50 | 237.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 236.42 | 236.40 | 237.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 11:15:00 | 236.42 | 236.40 | 237.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 236.42 | 236.40 | 237.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:30:00 | 235.99 | 236.40 | 237.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 236.88 | 236.50 | 236.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 236.88 | 236.50 | 236.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 237.90 | 236.78 | 237.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 237.90 | 236.78 | 237.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 237.72 | 236.97 | 237.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 238.44 | 236.97 | 237.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 237.00 | 236.97 | 237.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 237.00 | 236.97 | 237.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 238.19 | 237.22 | 237.22 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 236.45 | 237.25 | 237.28 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 239.81 | 237.64 | 237.44 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 236.55 | 237.45 | 237.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 235.36 | 236.84 | 237.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 235.80 | 232.52 | 233.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 235.80 | 232.52 | 233.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 235.80 | 232.52 | 233.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 235.80 | 232.52 | 233.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 235.61 | 233.14 | 233.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 231.67 | 233.14 | 233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 233.61 | 233.29 | 233.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 233.61 | 233.29 | 233.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 231.51 | 232.93 | 233.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:00:00 | 230.98 | 232.54 | 233.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 231.00 | 229.00 | 228.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 231.00 | 229.00 | 228.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 232.03 | 229.61 | 229.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 233.77 | 234.31 | 232.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:45:00 | 234.05 | 234.31 | 232.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 232.98 | 233.91 | 232.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 232.99 | 233.91 | 232.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 232.78 | 233.68 | 232.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 232.41 | 233.68 | 232.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 232.51 | 233.45 | 232.64 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 229.26 | 231.92 | 232.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 227.85 | 230.60 | 231.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 228.60 | 228.57 | 229.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 230.00 | 228.57 | 229.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 238.22 | 230.50 | 230.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 238.22 | 230.50 | 230.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 236.00 | 231.60 | 231.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 239.21 | 235.63 | 233.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 261.75 | 262.57 | 258.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 261.75 | 262.57 | 258.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 260.80 | 262.39 | 260.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 261.22 | 262.39 | 260.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 259.10 | 261.73 | 260.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 259.20 | 261.73 | 260.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 258.71 | 261.13 | 260.60 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 256.20 | 259.61 | 259.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 254.63 | 257.75 | 258.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 261.54 | 257.70 | 258.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 261.54 | 257.70 | 258.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 261.54 | 257.70 | 258.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 260.51 | 257.70 | 258.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 260.39 | 258.24 | 258.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 261.07 | 258.24 | 258.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 261.80 | 258.95 | 258.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 264.50 | 260.68 | 259.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 278.46 | 279.53 | 274.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 278.46 | 279.53 | 274.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 275.33 | 278.09 | 275.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 275.33 | 278.09 | 275.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 281.00 | 278.67 | 275.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 284.50 | 278.67 | 275.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 287.53 | 280.09 | 276.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 284.82 | 279.53 | 277.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:45:00 | 283.73 | 280.13 | 278.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 278.09 | 279.72 | 278.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 278.09 | 279.72 | 278.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 278.00 | 279.38 | 278.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:45:00 | 276.70 | 279.38 | 278.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 278.58 | 279.22 | 278.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 282.57 | 280.29 | 278.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 277.90 | 281.30 | 281.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 277.90 | 281.30 | 281.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 14:15:00 | 276.70 | 279.81 | 280.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 10:15:00 | 280.50 | 279.35 | 280.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 280.50 | 279.35 | 280.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 280.50 | 279.35 | 280.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 280.50 | 279.35 | 280.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 282.00 | 279.88 | 280.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 282.60 | 279.88 | 280.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 280.80 | 280.06 | 280.39 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 283.60 | 280.86 | 280.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 284.50 | 281.58 | 281.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 281.00 | 281.91 | 281.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 281.00 | 281.91 | 281.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 281.00 | 281.91 | 281.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 281.50 | 281.91 | 281.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 281.00 | 281.73 | 281.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 281.30 | 281.73 | 281.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 281.40 | 281.66 | 281.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 282.95 | 281.66 | 281.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 279.45 | 282.32 | 282.27 | SL hit (close<static) qty=1.00 sl=280.75 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 273.25 | 280.51 | 281.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 272.75 | 278.95 | 280.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 274.80 | 273.49 | 276.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 274.80 | 273.49 | 276.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 274.60 | 273.71 | 276.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 274.05 | 273.71 | 276.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 273.80 | 272.63 | 274.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 273.80 | 272.63 | 274.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 275.35 | 273.18 | 274.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 275.35 | 273.18 | 274.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 272.70 | 273.08 | 274.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 271.85 | 272.50 | 274.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:30:00 | 272.10 | 269.99 | 271.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 267.45 | 265.56 | 265.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 267.45 | 265.56 | 265.44 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 264.25 | 265.35 | 265.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 262.85 | 264.85 | 265.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 266.20 | 264.33 | 264.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 266.20 | 264.33 | 264.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 266.20 | 264.33 | 264.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 266.20 | 264.33 | 264.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 264.60 | 264.38 | 264.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 264.10 | 264.33 | 264.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 264.15 | 264.31 | 264.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 263.80 | 264.21 | 264.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:00:00 | 263.65 | 263.97 | 264.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 262.10 | 262.11 | 263.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 263.45 | 262.11 | 263.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 262.95 | 262.21 | 262.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 262.95 | 262.21 | 262.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 261.65 | 262.10 | 262.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 261.30 | 261.94 | 262.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 265.90 | 262.34 | 262.66 | SL hit (close>static) qty=1.00 sl=263.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 261.55 | 259.24 | 259.21 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 258.55 | 259.75 | 259.77 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 261.90 | 260.16 | 259.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 11:15:00 | 263.20 | 261.81 | 261.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 12:15:00 | 261.40 | 261.73 | 261.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 261.40 | 261.73 | 261.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 261.85 | 261.76 | 261.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 261.85 | 261.76 | 261.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 260.20 | 261.44 | 261.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 260.20 | 261.44 | 261.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 257.90 | 260.74 | 260.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 254.15 | 259.42 | 260.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 257.40 | 257.29 | 258.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 257.00 | 257.29 | 258.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 258.80 | 257.59 | 258.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 258.85 | 257.59 | 258.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 257.80 | 257.63 | 258.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 261.80 | 257.63 | 258.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 261.85 | 258.48 | 258.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 262.00 | 258.48 | 258.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 262.25 | 259.23 | 259.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 262.25 | 259.23 | 259.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 261.40 | 259.67 | 259.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 262.45 | 260.22 | 259.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 12:15:00 | 262.15 | 262.44 | 261.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 13:00:00 | 262.15 | 262.44 | 261.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 262.10 | 262.37 | 261.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 262.10 | 262.37 | 261.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 261.70 | 262.26 | 261.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 263.30 | 262.26 | 261.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 263.10 | 263.61 | 262.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 260.65 | 263.02 | 262.44 | SL hit (close<static) qty=1.00 sl=261.05 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 260.95 | 261.91 | 262.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 259.25 | 261.25 | 261.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 256.15 | 255.77 | 257.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 15:00:00 | 256.15 | 255.77 | 257.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 251.00 | 250.86 | 252.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:15:00 | 246.40 | 249.38 | 251.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 247.05 | 244.68 | 244.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 247.05 | 244.68 | 244.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 249.30 | 245.90 | 245.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 245.65 | 246.67 | 245.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 245.65 | 246.67 | 245.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 245.65 | 246.67 | 245.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 247.00 | 246.67 | 245.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 245.40 | 246.41 | 245.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 245.40 | 246.41 | 245.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 244.60 | 246.05 | 245.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 244.60 | 246.05 | 245.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 243.70 | 245.58 | 245.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 242.90 | 245.04 | 245.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 241.80 | 240.71 | 242.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 241.80 | 240.71 | 242.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 241.80 | 240.71 | 242.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 238.32 | 240.33 | 241.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 238.36 | 239.16 | 240.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 237.94 | 236.43 | 237.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 239.30 | 236.01 | 236.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 239.30 | 236.01 | 236.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 240.46 | 236.90 | 236.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 236.59 | 237.65 | 237.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 236.59 | 237.65 | 237.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 236.59 | 237.65 | 237.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 236.59 | 237.65 | 237.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 237.71 | 237.66 | 237.07 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 234.61 | 236.60 | 236.81 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 238.65 | 236.86 | 236.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 240.89 | 237.66 | 237.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 243.75 | 243.82 | 241.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:30:00 | 243.90 | 243.82 | 241.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 241.96 | 243.34 | 241.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 241.28 | 243.34 | 241.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 240.98 | 242.87 | 241.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 240.98 | 242.87 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 241.48 | 242.59 | 241.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 241.90 | 242.05 | 241.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 240.00 | 241.12 | 241.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 240.00 | 241.12 | 241.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 237.00 | 240.11 | 240.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 238.01 | 237.65 | 238.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 238.01 | 237.65 | 238.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 238.00 | 237.72 | 238.58 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 242.29 | 239.11 | 238.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 10:15:00 | 247.30 | 242.37 | 240.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 248.65 | 249.55 | 246.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:15:00 | 247.95 | 249.55 | 246.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 251.66 | 249.73 | 247.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:45:00 | 254.28 | 250.51 | 248.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 253.53 | 251.85 | 249.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 252.94 | 253.87 | 252.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:00:00 | 253.33 | 253.76 | 252.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 253.30 | 253.63 | 252.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 251.37 | 253.63 | 252.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 248.64 | 252.63 | 252.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 247.66 | 252.63 | 252.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 249.85 | 252.07 | 252.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 249.85 | 252.07 | 252.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 11:15:00 | 248.42 | 251.34 | 251.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 251.45 | 251.17 | 251.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 13:15:00 | 251.45 | 251.17 | 251.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 251.45 | 251.17 | 251.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 251.45 | 251.17 | 251.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 250.00 | 250.93 | 251.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 250.00 | 250.93 | 251.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 251.20 | 250.74 | 251.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 252.25 | 250.74 | 251.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 255.70 | 251.73 | 251.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 255.70 | 251.73 | 251.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 255.50 | 252.49 | 252.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 257.75 | 253.54 | 252.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 271.15 | 271.65 | 268.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 271.15 | 271.65 | 268.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 269.65 | 271.74 | 269.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 269.70 | 271.74 | 269.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 269.25 | 271.24 | 269.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 269.60 | 271.24 | 269.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 268.80 | 270.75 | 269.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:45:00 | 268.55 | 270.75 | 269.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 269.00 | 270.20 | 269.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 271.25 | 270.20 | 269.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 268.95 | 269.95 | 269.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:15:00 | 266.55 | 269.95 | 269.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 262.55 | 268.47 | 268.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 261.15 | 267.01 | 268.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 262.95 | 262.60 | 265.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 262.95 | 262.60 | 265.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 260.65 | 262.21 | 264.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 260.35 | 261.67 | 264.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 247.33 | 254.91 | 259.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 252.40 | 251.38 | 256.11 | SL hit (close>ema200) qty=0.50 sl=251.38 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 262.45 | 257.67 | 257.15 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 257.50 | 258.58 | 258.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 254.70 | 257.47 | 258.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 256.85 | 256.65 | 257.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:45:00 | 256.45 | 256.65 | 257.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 257.15 | 256.75 | 257.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 253.45 | 256.75 | 257.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 250.50 | 255.81 | 256.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 240.78 | 244.07 | 246.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 247.50 | 244.06 | 245.96 | SL hit (close>ema200) qty=0.50 sl=244.06 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 252.60 | 248.09 | 247.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 254.80 | 250.46 | 248.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 258.00 | 258.86 | 255.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:30:00 | 258.70 | 258.86 | 255.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 258.10 | 259.67 | 257.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 258.10 | 259.67 | 257.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 250.00 | 257.74 | 256.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 249.55 | 257.74 | 256.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 248.30 | 255.85 | 256.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 245.40 | 253.76 | 255.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 248.15 | 246.84 | 250.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 248.15 | 246.84 | 250.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 249.10 | 247.29 | 250.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 251.20 | 247.29 | 250.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 249.55 | 247.74 | 249.99 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 259.35 | 252.16 | 251.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 260.60 | 253.85 | 252.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 260.80 | 261.23 | 257.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:45:00 | 260.30 | 261.23 | 257.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 257.50 | 259.63 | 258.32 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 254.90 | 257.38 | 257.51 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 260.65 | 257.38 | 257.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 261.00 | 258.11 | 257.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 11:15:00 | 260.55 | 260.72 | 259.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:45:00 | 260.50 | 260.72 | 259.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 259.20 | 260.37 | 259.67 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 253.65 | 258.19 | 258.78 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 259.90 | 257.69 | 257.54 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 253.55 | 257.23 | 257.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 252.65 | 256.32 | 256.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 14:15:00 | 260.00 | 255.03 | 255.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 260.00 | 255.03 | 255.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 260.00 | 255.03 | 255.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 260.00 | 255.03 | 255.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 258.55 | 255.73 | 256.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 254.80 | 255.73 | 256.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 256.90 | 255.93 | 255.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 263.05 | 256.68 | 256.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 263.05 | 256.68 | 256.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 264.85 | 259.40 | 257.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 263.25 | 263.66 | 261.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 13:45:00 | 263.35 | 263.66 | 261.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 258.10 | 262.42 | 261.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 257.40 | 262.42 | 261.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 264.70 | 262.87 | 261.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 266.60 | 264.00 | 262.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 13:15:00 | 262.15 | 263.32 | 263.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 262.15 | 263.32 | 263.35 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 265.30 | 263.72 | 263.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 265.65 | 264.14 | 263.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 264.25 | 264.25 | 263.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:30:00 | 264.35 | 264.25 | 263.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 261.70 | 263.74 | 263.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 261.70 | 263.74 | 263.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 263.35 | 263.66 | 263.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 264.05 | 263.66 | 263.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 263.10 | 263.63 | 263.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 263.10 | 263.63 | 263.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 261.40 | 263.13 | 263.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 262.70 | 262.62 | 263.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 262.70 | 262.62 | 263.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 262.70 | 262.62 | 263.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 262.40 | 262.62 | 263.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 260.50 | 252.01 | 253.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 260.50 | 252.01 | 253.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 256.40 | 252.89 | 253.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 254.30 | 253.21 | 254.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 258.00 | 254.39 | 254.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 258.00 | 254.39 | 254.29 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 246.00 | 254.45 | 254.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 243.25 | 252.21 | 253.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 245.20 | 244.39 | 248.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 245.20 | 244.39 | 248.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 248.20 | 245.43 | 246.96 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 252.00 | 248.55 | 248.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 257.95 | 250.35 | 249.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 262.25 | 262.96 | 258.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 12:30:00 | 262.30 | 262.96 | 258.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 260.25 | 262.23 | 259.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 260.25 | 262.23 | 259.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 259.50 | 261.69 | 259.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 257.80 | 261.69 | 259.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 260.85 | 261.52 | 259.29 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 15:15:00 | 256.95 | 258.39 | 258.44 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 262.05 | 259.12 | 258.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 10:15:00 | 264.45 | 260.19 | 259.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 261.80 | 263.37 | 262.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 261.80 | 263.37 | 262.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 261.80 | 263.37 | 262.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 261.80 | 263.37 | 262.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 262.25 | 263.15 | 262.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 263.35 | 263.15 | 262.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 263.15 | 263.15 | 262.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 265.15 | 263.41 | 262.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 268.25 | 262.50 | 262.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 256.35 | 265.36 | 264.98 | SL hit (close<static) qty=1.00 sl=257.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 254.45 | 263.18 | 264.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 252.40 | 261.02 | 262.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 257.15 | 255.48 | 258.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 257.15 | 255.48 | 258.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 258.65 | 256.12 | 258.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 258.35 | 256.12 | 258.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 259.80 | 256.85 | 258.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 259.50 | 256.85 | 258.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 259.30 | 257.34 | 258.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 258.00 | 257.34 | 258.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 268.00 | 259.58 | 259.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 268.00 | 259.58 | 259.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 13:15:00 | 271.45 | 267.74 | 265.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 14:15:00 | 268.85 | 270.96 | 268.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 268.85 | 270.96 | 268.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 268.85 | 270.96 | 268.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 268.85 | 270.96 | 268.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 267.85 | 270.33 | 268.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 275.75 | 270.33 | 268.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 274.55 | 275.95 | 276.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 274.55 | 275.95 | 276.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 11:15:00 | 272.00 | 275.16 | 275.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 273.55 | 273.19 | 274.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 273.55 | 273.19 | 274.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 273.55 | 273.19 | 274.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:45:00 | 272.30 | 273.04 | 273.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 273.50 | 270.58 | 270.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 13:15:00 | 273.50 | 270.58 | 270.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 278.70 | 272.46 | 271.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 304.75 | 305.10 | 299.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 304.75 | 305.10 | 299.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 300.40 | 303.31 | 301.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 300.45 | 303.31 | 301.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 299.80 | 302.61 | 300.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 299.70 | 302.61 | 300.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 305.05 | 302.88 | 301.38 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 299.80 | 301.05 | 301.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 298.15 | 300.47 | 300.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 298.35 | 298.13 | 299.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 298.35 | 298.13 | 299.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 298.35 | 298.13 | 299.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 303.20 | 298.13 | 299.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 305.20 | 299.54 | 299.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:45:00 | 305.70 | 299.54 | 299.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 11:15:00 | 302.40 | 300.11 | 300.09 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 299.75 | 300.04 | 300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 295.20 | 299.07 | 299.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 300.40 | 299.34 | 299.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 300.40 | 299.34 | 299.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 300.40 | 299.34 | 299.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 300.40 | 299.34 | 299.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 300.00 | 299.47 | 299.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 305.70 | 299.47 | 299.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 313.75 | 302.33 | 300.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 315.50 | 304.96 | 302.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 315.70 | 316.65 | 311.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:45:00 | 316.15 | 316.65 | 311.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 314.10 | 316.45 | 312.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 314.30 | 316.45 | 312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 313.80 | 317.01 | 315.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 313.80 | 317.01 | 315.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 314.95 | 316.60 | 315.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 323.55 | 316.60 | 315.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 316.80 | 317.12 | 315.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 317.00 | 319.85 | 318.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 316.80 | 318.88 | 318.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 315.60 | 317.82 | 317.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 315.60 | 317.82 | 317.95 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 318.80 | 317.84 | 317.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 321.80 | 318.63 | 318.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 12:15:00 | 321.20 | 321.92 | 320.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 12:15:00 | 321.20 | 321.92 | 320.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 321.20 | 321.92 | 320.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 321.20 | 321.92 | 320.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 322.30 | 321.99 | 320.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 14:45:00 | 327.85 | 323.20 | 321.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 11:45:00 | 227.00 | 2024-05-24 10:15:00 | 237.90 | STOP_HIT | 1.00 | 4.80% |
| BUY | retest2 | 2024-05-16 14:45:00 | 227.10 | 2024-05-24 10:15:00 | 237.90 | STOP_HIT | 1.00 | 4.76% |
| BUY | retest2 | 2024-05-17 10:45:00 | 229.25 | 2024-05-24 10:15:00 | 237.90 | STOP_HIT | 1.00 | 3.77% |
| SELL | retest2 | 2024-06-06 11:30:00 | 216.50 | 2024-06-07 09:15:00 | 217.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-06-06 13:45:00 | 216.95 | 2024-06-07 09:15:00 | 217.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-06-13 15:15:00 | 233.45 | 2024-06-18 09:15:00 | 229.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-06-14 09:30:00 | 233.05 | 2024-06-18 09:15:00 | 229.60 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-06-14 11:00:00 | 233.11 | 2024-06-18 09:15:00 | 229.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-06-14 12:45:00 | 233.16 | 2024-06-18 09:15:00 | 229.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-06-19 14:45:00 | 236.19 | 2024-06-20 14:15:00 | 231.52 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-07-16 11:00:00 | 308.97 | 2024-07-18 11:15:00 | 283.50 | STOP_HIT | 1.00 | -8.24% |
| BUY | retest1 | 2024-07-29 09:15:00 | 290.27 | 2024-07-29 13:15:00 | 285.28 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2024-07-29 10:45:00 | 289.90 | 2024-07-29 13:15:00 | 285.28 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-07-30 09:15:00 | 288.74 | 2024-08-01 11:15:00 | 286.05 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-07-30 10:00:00 | 292.74 | 2024-08-01 11:15:00 | 286.05 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-08-16 15:00:00 | 261.45 | 2024-08-19 09:15:00 | 264.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-19 09:15:00 | 261.10 | 2024-08-19 09:15:00 | 264.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-08-19 13:30:00 | 261.70 | 2024-08-20 09:15:00 | 271.35 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-08-19 14:00:00 | 261.60 | 2024-08-20 09:15:00 | 271.35 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-08-27 09:15:00 | 281.00 | 2024-08-29 14:15:00 | 274.65 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-10-01 09:15:00 | 283.90 | 2024-10-03 15:15:00 | 282.75 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-10-01 10:45:00 | 284.25 | 2024-10-03 15:15:00 | 282.75 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-10-16 09:15:00 | 267.75 | 2024-10-16 10:15:00 | 270.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-16 09:45:00 | 268.25 | 2024-10-16 10:15:00 | 270.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-10-21 09:15:00 | 262.00 | 2024-10-22 10:15:00 | 248.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 262.00 | 2024-10-23 09:15:00 | 235.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-31 10:15:00 | 256.45 | 2024-11-08 10:15:00 | 253.75 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-10-31 14:30:00 | 254.80 | 2024-11-08 10:15:00 | 253.75 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-11-05 09:45:00 | 255.75 | 2024-11-08 10:15:00 | 253.75 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-11-18 14:15:00 | 235.15 | 2024-11-21 09:15:00 | 241.25 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2024-12-04 09:15:00 | 265.80 | 2024-12-05 09:15:00 | 262.85 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-06 10:30:00 | 268.05 | 2024-12-10 12:15:00 | 263.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-12-09 09:15:00 | 269.35 | 2024-12-10 12:15:00 | 263.30 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-12-10 11:00:00 | 266.70 | 2024-12-10 12:15:00 | 263.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-12-17 11:45:00 | 255.90 | 2024-12-19 09:15:00 | 243.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 255.90 | 2024-12-19 09:15:00 | 253.45 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-12-17 12:45:00 | 255.90 | 2024-12-19 09:15:00 | 243.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:45:00 | 255.90 | 2024-12-19 09:15:00 | 253.45 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-12-30 15:00:00 | 233.75 | 2024-12-31 15:15:00 | 241.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-01-08 15:15:00 | 232.47 | 2025-01-09 12:15:00 | 238.67 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-01-15 15:15:00 | 228.63 | 2025-01-20 14:15:00 | 232.30 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-16 13:45:00 | 229.41 | 2025-01-20 14:15:00 | 232.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-01-17 09:30:00 | 228.85 | 2025-01-20 14:15:00 | 232.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-17 10:45:00 | 228.31 | 2025-01-20 14:15:00 | 232.30 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-02-01 15:15:00 | 219.50 | 2025-02-03 09:15:00 | 210.18 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-03-03 11:00:00 | 200.00 | 2025-03-04 13:15:00 | 211.82 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest1 | 2025-03-06 09:30:00 | 225.71 | 2025-03-06 12:15:00 | 237.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-06 09:30:00 | 225.71 | 2025-03-06 15:15:00 | 227.35 | STOP_HIT | 0.50 | 0.73% |
| BUY | retest2 | 2025-03-07 14:15:00 | 227.40 | 2025-03-10 10:15:00 | 223.90 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-03-17 09:15:00 | 228.33 | 2025-03-21 09:15:00 | 251.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 11:30:00 | 244.40 | 2025-04-03 09:15:00 | 247.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-03-28 15:15:00 | 242.50 | 2025-04-03 09:15:00 | 247.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-04-01 09:45:00 | 244.59 | 2025-04-03 09:15:00 | 247.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-01 10:15:00 | 242.94 | 2025-04-03 09:15:00 | 247.60 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-04-23 09:15:00 | 248.19 | 2025-04-23 09:15:00 | 242.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-04-24 09:45:00 | 248.20 | 2025-04-24 12:15:00 | 243.63 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-24 10:45:00 | 246.50 | 2025-04-24 12:15:00 | 243.63 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2025-05-02 12:15:00 | 226.15 | 2025-05-08 09:15:00 | 226.35 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest1 | 2025-05-05 12:30:00 | 226.93 | 2025-05-08 09:15:00 | 226.35 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-05-06 11:30:00 | 222.25 | 2025-05-08 11:15:00 | 226.73 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-05-06 12:15:00 | 222.82 | 2025-05-08 11:15:00 | 226.73 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-05-07 13:15:00 | 222.27 | 2025-05-08 11:15:00 | 226.73 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-20 09:15:00 | 254.21 | 2025-05-23 09:15:00 | 237.80 | STOP_HIT | 1.00 | -6.46% |
| SELL | retest1 | 2025-06-17 11:45:00 | 230.16 | 2025-06-23 09:15:00 | 225.29 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-06-18 11:45:00 | 227.15 | 2025-06-23 13:15:00 | 227.48 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-23 12:00:00 | 227.30 | 2025-06-23 13:15:00 | 227.48 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-25 09:15:00 | 229.56 | 2025-06-26 12:15:00 | 226.94 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-06-25 14:30:00 | 228.26 | 2025-06-26 12:15:00 | 226.94 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-26 09:15:00 | 230.43 | 2025-06-26 12:15:00 | 226.94 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-06-30 09:15:00 | 232.42 | 2025-07-01 09:15:00 | 226.82 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-07-04 13:15:00 | 226.01 | 2025-07-04 14:15:00 | 227.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-04 14:15:00 | 226.47 | 2025-07-04 14:15:00 | 227.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-07-07 09:15:00 | 226.36 | 2025-07-11 09:15:00 | 226.39 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-08 09:45:00 | 226.13 | 2025-07-11 09:15:00 | 226.39 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-08 12:00:00 | 224.11 | 2025-07-11 09:15:00 | 226.39 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-09 09:45:00 | 224.38 | 2025-07-11 10:15:00 | 227.05 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-10 10:45:00 | 224.39 | 2025-07-11 10:15:00 | 227.05 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-15 12:30:00 | 228.60 | 2025-07-21 09:15:00 | 233.62 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-08-06 09:15:00 | 241.00 | 2025-08-06 09:15:00 | 238.37 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-08-11 09:30:00 | 230.84 | 2025-08-12 11:15:00 | 238.52 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-08-11 11:30:00 | 230.53 | 2025-08-12 11:15:00 | 238.52 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-08-28 13:00:00 | 230.98 | 2025-09-01 15:15:00 | 231.00 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-09-26 09:15:00 | 284.50 | 2025-10-01 12:15:00 | 277.90 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-09-26 09:45:00 | 287.53 | 2025-10-01 12:15:00 | 277.90 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-09-29 10:15:00 | 284.82 | 2025-10-01 12:15:00 | 277.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-09-29 10:45:00 | 283.73 | 2025-10-01 12:15:00 | 277.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-09-29 14:45:00 | 282.57 | 2025-10-01 12:15:00 | 277.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-10-06 15:15:00 | 282.95 | 2025-10-08 09:15:00 | 279.45 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-10 13:30:00 | 271.85 | 2025-10-20 15:15:00 | 267.45 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2025-10-14 09:30:00 | 272.10 | 2025-10-20 15:15:00 | 267.45 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2025-10-24 12:00:00 | 264.10 | 2025-10-29 09:15:00 | 265.90 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-24 13:30:00 | 264.15 | 2025-10-30 14:15:00 | 264.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-24 15:00:00 | 263.80 | 2025-10-31 14:15:00 | 261.55 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2025-10-27 10:00:00 | 263.65 | 2025-10-31 14:15:00 | 261.55 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-10-28 14:00:00 | 261.30 | 2025-10-31 14:15:00 | 261.55 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-10-29 11:45:00 | 260.90 | 2025-10-31 14:15:00 | 261.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-30 15:15:00 | 261.40 | 2025-10-31 14:15:00 | 261.55 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-11-12 09:15:00 | 263.30 | 2025-11-12 14:15:00 | 260.65 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-12 14:15:00 | 263.10 | 2025-11-12 14:15:00 | 260.65 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-21 14:15:00 | 246.40 | 2025-11-26 15:15:00 | 247.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-12-03 09:30:00 | 238.32 | 2025-12-09 12:15:00 | 239.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-12-04 09:15:00 | 238.36 | 2025-12-09 12:15:00 | 239.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-08 09:15:00 | 237.94 | 2025-12-09 12:15:00 | 239.30 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-16 11:45:00 | 241.90 | 2025-12-16 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-26 10:45:00 | 254.28 | 2025-12-31 10:15:00 | 249.85 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-29 09:30:00 | 253.53 | 2025-12-31 10:15:00 | 249.85 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-30 13:00:00 | 252.94 | 2025-12-31 10:15:00 | 249.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-30 14:00:00 | 253.33 | 2025-12-31 10:15:00 | 249.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-09 11:30:00 | 260.35 | 2026-01-12 09:15:00 | 247.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 260.35 | 2026-01-12 13:15:00 | 252.40 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-01-20 09:15:00 | 253.45 | 2026-01-27 13:15:00 | 240.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 253.45 | 2026-01-28 09:15:00 | 247.50 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2026-01-21 09:15:00 | 250.50 | 2026-01-28 12:15:00 | 252.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-16 09:15:00 | 254.80 | 2026-02-17 14:15:00 | 263.05 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-02-17 09:45:00 | 256.90 | 2026-02-17 14:15:00 | 263.05 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-02-23 09:15:00 | 266.60 | 2026-02-24 13:15:00 | 262.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2026-02-25 14:15:00 | 264.05 | 2026-02-25 15:15:00 | 263.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-03-05 11:45:00 | 254.30 | 2026-03-06 09:15:00 | 258.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-03-19 10:30:00 | 265.15 | 2026-03-23 09:15:00 | 256.35 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-20 09:15:00 | 268.25 | 2026-03-23 09:15:00 | 256.35 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2026-03-24 15:15:00 | 258.00 | 2026-03-25 09:15:00 | 268.00 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2026-04-01 09:15:00 | 275.75 | 2026-04-07 10:15:00 | 274.55 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-04-08 12:45:00 | 272.30 | 2026-04-13 13:15:00 | 273.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-04-30 11:15:00 | 323.55 | 2026-05-05 13:15:00 | 315.60 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-04-30 13:30:00 | 316.80 | 2026-05-05 13:15:00 | 315.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2026-05-05 11:00:00 | 317.00 | 2026-05-05 13:15:00 | 315.60 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-05-05 12:15:00 | 316.80 | 2026-05-05 13:15:00 | 315.60 | STOP_HIT | 1.00 | -0.38% |
