# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 243.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 108 |
| ALERT2 | 107 |
| ALERT2_SKIP | 57 |
| ALERT3 | 319 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 121 |
| PARTIAL | 30 |
| TARGET_HIT | 4 |
| STOP_HIT | 126 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 160 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 94
- **Target hits / Stop hits / Partials:** 4 / 126 / 30
- **Avg / median % per leg:** 0.26% / -0.97%
- **Sum % (uncompounded):** 41.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 10 | 17.2% | 3 | 52 | 3 | -1.03% | -59.9% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 0 | 9 | 3 | 0.16% | 2.0% |
| BUY @ 3rd Alert (retest2) | 46 | 4 | 8.7% | 3 | 43 | 0 | -1.35% | -61.9% |
| SELL (all) | 102 | 56 | 54.9% | 1 | 74 | 27 | 0.99% | 101.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 102 | 56 | 54.9% | 1 | 74 | 27 | 0.99% | 101.5% |
| retest1 (combined) | 12 | 6 | 50.0% | 0 | 9 | 3 | 0.16% | 2.0% |
| retest2 (combined) | 148 | 60 | 40.5% | 4 | 117 | 27 | 0.27% | 39.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 271.95 | 266.65 | 266.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 273.63 | 269.75 | 267.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 272.05 | 274.71 | 272.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 272.05 | 274.71 | 272.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 272.05 | 274.71 | 272.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 272.05 | 274.71 | 272.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 270.00 | 273.77 | 272.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 269.80 | 273.77 | 272.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 275.15 | 275.04 | 273.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:00:00 | 275.15 | 275.04 | 273.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 276.00 | 275.27 | 274.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 278.23 | 275.27 | 274.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 272.60 | 275.08 | 274.47 | SL hit (close<static) qty=1.00 sl=273.85 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 272.70 | 274.09 | 274.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 269.05 | 272.53 | 273.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 276.83 | 272.79 | 273.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 11:15:00 | 276.83 | 272.79 | 273.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 276.83 | 272.79 | 273.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 276.83 | 272.79 | 273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 280.18 | 274.26 | 273.90 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 272.05 | 273.89 | 274.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 270.00 | 272.60 | 273.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 273.68 | 272.36 | 272.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 273.68 | 272.36 | 272.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 273.68 | 272.36 | 272.81 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 278.00 | 273.71 | 273.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 281.98 | 275.37 | 274.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 10:15:00 | 277.95 | 278.16 | 276.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:30:00 | 277.65 | 278.16 | 276.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 277.80 | 277.97 | 276.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 276.52 | 277.97 | 276.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 276.50 | 277.60 | 276.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 278.23 | 277.60 | 276.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 275.65 | 277.21 | 276.51 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 11:15:00 | 273.45 | 275.77 | 275.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 268.50 | 273.03 | 274.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 273.10 | 272.51 | 273.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 273.10 | 272.51 | 273.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 269.02 | 271.81 | 273.15 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 283.35 | 275.15 | 274.37 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 246.95 | 270.23 | 273.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 243.15 | 264.82 | 270.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 253.23 | 251.55 | 258.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 253.23 | 251.55 | 258.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 261.70 | 254.34 | 257.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 262.48 | 254.34 | 257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 265.50 | 256.57 | 258.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 265.50 | 256.57 | 258.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 262.88 | 259.98 | 259.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 263.88 | 260.76 | 260.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 279.02 | 279.05 | 273.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 279.63 | 279.05 | 273.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 277.48 | 278.73 | 274.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:45:00 | 281.60 | 279.09 | 275.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 279.70 | 279.12 | 275.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:00:00 | 280.00 | 279.30 | 275.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 280.23 | 278.13 | 276.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 278.50 | 278.18 | 276.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:45:00 | 279.70 | 277.57 | 276.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:30:00 | 279.50 | 277.93 | 277.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 280.00 | 278.29 | 277.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:30:00 | 279.85 | 279.23 | 277.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 279.95 | 279.66 | 278.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 285.68 | 279.66 | 278.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 280.68 | 280.47 | 279.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:30:00 | 281.00 | 280.50 | 279.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 274.18 | 278.74 | 279.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 274.18 | 278.74 | 279.11 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 282.13 | 279.56 | 279.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 286.65 | 281.53 | 280.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 13:15:00 | 281.63 | 281.99 | 280.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 13:15:00 | 281.63 | 281.99 | 280.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 281.63 | 281.99 | 280.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 281.63 | 281.99 | 280.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 281.40 | 281.87 | 280.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 282.50 | 281.87 | 280.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 11:45:00 | 282.58 | 284.17 | 284.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 282.10 | 284.17 | 284.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 281.50 | 283.64 | 283.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 281.50 | 283.64 | 283.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 279.15 | 282.74 | 283.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 278.33 | 277.94 | 279.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 10:00:00 | 278.33 | 277.94 | 279.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 275.90 | 273.51 | 275.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 275.33 | 273.51 | 275.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 274.10 | 273.63 | 275.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 272.73 | 273.63 | 275.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:00:00 | 273.52 | 273.59 | 275.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:45:00 | 273.05 | 272.98 | 274.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 273.15 | 272.77 | 273.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 271.25 | 272.46 | 273.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:15:00 | 270.68 | 272.46 | 273.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:00:00 | 269.55 | 271.88 | 273.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 271.18 | 271.55 | 272.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 278.40 | 273.07 | 273.07 | SL hit (close>static) qty=1.00 sl=275.93 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 276.75 | 273.81 | 273.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 280.50 | 277.01 | 275.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 276.00 | 277.01 | 275.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 12:15:00 | 276.00 | 277.01 | 275.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 276.00 | 277.01 | 275.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:00:00 | 276.00 | 277.01 | 275.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 276.25 | 276.86 | 275.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 276.25 | 276.86 | 275.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 276.58 | 276.80 | 275.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:15:00 | 277.00 | 276.80 | 275.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 277.00 | 276.84 | 275.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 275.65 | 276.84 | 275.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 274.27 | 276.33 | 275.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 274.70 | 276.33 | 275.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 278.50 | 276.76 | 276.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 275.02 | 276.76 | 276.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 276.23 | 277.19 | 276.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 276.23 | 277.19 | 276.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 276.48 | 277.04 | 276.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 276.48 | 277.04 | 276.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 276.40 | 276.92 | 276.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 279.43 | 276.92 | 276.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 281.45 | 283.17 | 281.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 281.45 | 283.17 | 281.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 279.00 | 282.34 | 281.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 279.00 | 282.34 | 281.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 279.50 | 281.77 | 281.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 285.75 | 281.77 | 281.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 278.15 | 280.65 | 280.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 278.15 | 280.65 | 280.68 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 282.27 | 280.46 | 280.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 284.18 | 281.91 | 281.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 282.20 | 282.23 | 281.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 282.20 | 282.23 | 281.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 282.20 | 282.23 | 281.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 280.98 | 282.23 | 281.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 283.60 | 282.51 | 281.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:30:00 | 283.00 | 282.51 | 281.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 281.73 | 282.31 | 281.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 279.33 | 282.31 | 281.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 280.98 | 282.05 | 281.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 279.88 | 282.05 | 281.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 280.35 | 281.71 | 281.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 279.63 | 281.71 | 281.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 279.77 | 281.32 | 281.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 278.40 | 280.74 | 281.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 281.05 | 279.89 | 280.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 281.05 | 279.89 | 280.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 281.05 | 279.89 | 280.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 281.05 | 279.89 | 280.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 283.95 | 280.70 | 280.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:45:00 | 285.20 | 280.70 | 280.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 283.77 | 281.31 | 281.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 291.50 | 283.77 | 282.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 281.65 | 284.21 | 283.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 14:15:00 | 281.65 | 284.21 | 283.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 281.65 | 284.21 | 283.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 281.65 | 284.21 | 283.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 282.50 | 283.87 | 283.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 278.65 | 283.87 | 283.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 274.25 | 281.94 | 282.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 267.25 | 279.00 | 280.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 14:15:00 | 275.08 | 274.27 | 277.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 15:00:00 | 275.08 | 274.27 | 277.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 276.20 | 274.69 | 277.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 276.10 | 274.69 | 277.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 276.75 | 275.10 | 277.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 277.52 | 275.10 | 277.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 275.60 | 275.20 | 277.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 276.85 | 275.20 | 277.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 276.65 | 275.16 | 276.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 276.65 | 275.16 | 276.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 274.50 | 275.03 | 276.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 273.30 | 275.03 | 276.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 278.00 | 274.18 | 274.63 | SL hit (close>static) qty=1.00 sl=277.25 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 278.80 | 275.10 | 275.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 286.63 | 280.14 | 278.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 322.00 | 322.20 | 317.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:15:00 | 328.58 | 322.20 | 317.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:45:00 | 324.60 | 323.27 | 319.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 13:00:00 | 324.85 | 323.59 | 319.64 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 14:15:00 | 324.68 | 323.68 | 320.05 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 325.95 | 324.40 | 321.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 317.60 | 324.40 | 321.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 312.50 | 322.02 | 320.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 312.50 | 322.02 | 320.50 | SL hit (close<ema400) qty=1.00 sl=320.50 alert=retest1 |

### Cycle 20 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 312.40 | 318.58 | 319.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 304.68 | 311.92 | 314.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 310.50 | 306.86 | 310.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 11:15:00 | 310.50 | 306.86 | 310.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 310.50 | 306.86 | 310.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 311.50 | 306.86 | 310.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 310.50 | 307.59 | 310.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 310.50 | 307.59 | 310.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 320.00 | 310.07 | 311.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 319.50 | 310.07 | 311.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 319.05 | 311.87 | 312.23 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 319.10 | 313.31 | 312.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 324.08 | 315.45 | 313.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 15:15:00 | 315.60 | 316.45 | 315.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:15:00 | 318.77 | 316.45 | 315.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:45:00 | 318.95 | 317.13 | 315.58 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 13:15:00 | 334.71 | 321.64 | 318.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 13:15:00 | 334.90 | 321.64 | 318.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 322.52 | 323.64 | 320.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 322.52 | 323.64 | 320.12 | SL hit (close<ema200) qty=0.50 sl=323.64 alert=retest1 |

### Cycle 22 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 316.98 | 319.61 | 319.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 315.00 | 318.69 | 319.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 308.27 | 308.07 | 312.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:30:00 | 307.75 | 308.07 | 312.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 310.02 | 308.98 | 311.56 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 314.98 | 312.51 | 312.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 323.20 | 314.64 | 313.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 321.77 | 322.45 | 320.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:15:00 | 325.23 | 322.45 | 320.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:15:00 | 341.49 | 330.83 | 326.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 338.80 | 339.44 | 333.55 | SL hit (close<ema200) qty=0.50 sl=339.44 alert=retest1 |

### Cycle 24 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 342.95 | 345.25 | 345.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 342.18 | 344.64 | 344.99 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 347.98 | 345.30 | 345.26 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 344.23 | 345.11 | 345.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 337.83 | 343.65 | 344.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 11:15:00 | 340.13 | 339.86 | 341.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:30:00 | 338.78 | 339.86 | 341.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 341.30 | 340.15 | 341.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 341.88 | 340.15 | 341.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 340.70 | 340.26 | 341.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 341.10 | 340.26 | 341.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 339.55 | 340.12 | 341.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 342.75 | 340.12 | 341.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 345.48 | 341.21 | 341.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 345.50 | 341.21 | 341.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 345.40 | 342.05 | 342.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 345.68 | 342.05 | 342.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 342.28 | 342.09 | 342.07 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 341.25 | 341.92 | 341.99 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 343.03 | 342.15 | 342.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 15:15:00 | 343.50 | 342.59 | 342.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 350.35 | 355.55 | 354.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 350.35 | 355.55 | 354.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 350.35 | 355.55 | 354.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 350.35 | 355.55 | 354.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 348.20 | 354.08 | 353.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 348.20 | 354.08 | 353.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 348.63 | 352.74 | 353.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 347.83 | 351.76 | 352.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 348.95 | 347.64 | 350.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:00:00 | 348.95 | 347.64 | 350.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 346.95 | 344.19 | 346.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 347.63 | 344.19 | 346.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 346.93 | 344.74 | 346.86 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 352.73 | 348.59 | 348.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 354.00 | 349.67 | 348.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 348.45 | 349.43 | 348.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 348.45 | 349.43 | 348.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 348.45 | 349.43 | 348.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 348.45 | 349.43 | 348.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 347.60 | 349.06 | 348.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:15:00 | 346.30 | 349.06 | 348.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 343.75 | 348.00 | 348.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 342.45 | 346.21 | 347.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 347.78 | 345.13 | 346.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 347.78 | 345.13 | 346.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 347.78 | 345.13 | 346.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 347.70 | 345.13 | 346.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 351.85 | 346.47 | 346.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 351.85 | 346.47 | 346.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 353.00 | 348.13 | 347.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 354.85 | 351.27 | 349.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 357.65 | 358.26 | 355.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:15:00 | 357.68 | 358.26 | 355.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 355.25 | 357.88 | 355.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 355.48 | 357.88 | 355.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 355.23 | 357.35 | 355.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 355.23 | 357.35 | 355.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 355.40 | 356.96 | 355.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:45:00 | 356.10 | 356.83 | 355.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:45:00 | 356.98 | 357.05 | 356.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 14:45:00 | 357.25 | 357.17 | 356.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 354.43 | 356.50 | 356.39 | SL hit (close<static) qty=1.00 sl=355.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 352.28 | 355.66 | 356.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 348.15 | 354.16 | 355.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 355.50 | 351.55 | 353.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 355.50 | 351.55 | 353.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 355.50 | 351.55 | 353.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 355.15 | 351.55 | 353.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 356.80 | 352.60 | 353.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 356.15 | 352.60 | 353.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 355.45 | 354.24 | 354.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 360.15 | 355.69 | 354.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 361.95 | 368.04 | 364.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 361.95 | 368.04 | 364.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 361.95 | 368.04 | 364.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 361.95 | 368.04 | 364.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 362.05 | 366.84 | 363.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 360.23 | 366.84 | 363.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 371.55 | 372.82 | 371.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 371.55 | 372.82 | 371.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 372.00 | 372.65 | 371.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 367.75 | 372.65 | 371.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 368.10 | 371.74 | 370.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 367.63 | 371.74 | 370.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 367.70 | 370.93 | 370.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 368.05 | 370.93 | 370.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 367.50 | 370.04 | 370.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 357.50 | 366.84 | 368.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 366.93 | 363.40 | 365.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 366.93 | 363.40 | 365.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 366.93 | 363.40 | 365.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:45:00 | 368.18 | 363.40 | 365.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 365.73 | 363.87 | 365.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 364.00 | 364.21 | 365.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:45:00 | 363.40 | 363.62 | 364.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 14:15:00 | 345.80 | 358.10 | 361.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 14:15:00 | 345.23 | 358.10 | 361.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 363.30 | 358.24 | 360.89 | SL hit (close>ema200) qty=0.50 sl=358.24 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 372.20 | 364.22 | 363.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 373.75 | 366.12 | 364.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 369.70 | 370.83 | 368.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 369.70 | 370.83 | 368.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 369.70 | 370.83 | 368.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 369.70 | 370.83 | 368.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 369.75 | 370.62 | 368.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 368.10 | 370.62 | 368.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 364.90 | 369.47 | 368.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 365.00 | 369.47 | 368.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 363.00 | 368.18 | 367.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 363.00 | 368.18 | 367.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 363.25 | 366.76 | 366.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 361.05 | 365.11 | 366.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 10:15:00 | 359.05 | 357.35 | 360.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 10:15:00 | 359.05 | 357.35 | 360.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 359.05 | 357.35 | 360.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:45:00 | 360.20 | 357.35 | 360.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 358.75 | 357.63 | 360.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:45:00 | 360.10 | 357.63 | 360.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 360.50 | 358.20 | 360.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:30:00 | 360.00 | 358.20 | 360.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 360.15 | 358.59 | 360.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:45:00 | 360.30 | 358.59 | 360.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 360.90 | 359.05 | 360.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:30:00 | 360.90 | 359.05 | 360.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 360.00 | 359.24 | 360.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:45:00 | 358.50 | 360.19 | 360.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 366.30 | 361.41 | 360.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 366.30 | 361.41 | 360.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-17 12:15:00 | 370.60 | 365.73 | 363.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 15:15:00 | 364.20 | 366.52 | 364.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 15:15:00 | 364.20 | 366.52 | 364.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 364.20 | 366.52 | 364.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 355.70 | 366.52 | 364.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 364.05 | 366.03 | 364.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 365.20 | 366.11 | 364.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 365.20 | 365.58 | 365.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 347.80 | 361.96 | 363.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 347.80 | 361.96 | 363.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 343.60 | 358.29 | 361.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 10:15:00 | 343.30 | 343.14 | 350.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 11:00:00 | 343.30 | 343.14 | 350.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 305.10 | 303.16 | 306.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 313.15 | 303.16 | 306.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 316.30 | 305.79 | 307.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 316.30 | 305.79 | 307.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 312.85 | 307.20 | 308.13 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 314.45 | 309.68 | 309.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 316.80 | 313.57 | 311.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 309.45 | 313.79 | 312.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 309.45 | 313.79 | 312.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 309.45 | 313.79 | 312.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 309.45 | 313.79 | 312.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 308.55 | 312.74 | 312.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 308.55 | 312.74 | 312.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 310.10 | 311.66 | 311.79 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 317.20 | 312.42 | 312.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 326.80 | 317.72 | 314.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 322.00 | 324.64 | 321.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:00:00 | 322.00 | 324.64 | 321.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 319.30 | 323.57 | 321.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 319.30 | 323.57 | 321.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 320.45 | 322.95 | 321.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 319.35 | 322.95 | 321.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 319.00 | 322.16 | 320.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 319.00 | 322.16 | 320.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 314.80 | 319.53 | 319.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 311.70 | 317.97 | 319.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 308.65 | 307.21 | 310.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 308.65 | 307.21 | 310.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 308.65 | 307.21 | 310.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 308.60 | 307.21 | 310.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 309.95 | 307.76 | 310.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 307.40 | 307.85 | 309.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 292.03 | 303.21 | 306.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 295.80 | 294.17 | 299.53 | SL hit (close>ema200) qty=0.50 sl=294.17 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 304.10 | 299.45 | 299.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 307.35 | 302.89 | 301.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 13:15:00 | 304.10 | 304.54 | 302.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 14:00:00 | 304.10 | 304.54 | 302.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 301.05 | 303.84 | 302.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 301.05 | 303.84 | 302.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 298.15 | 302.70 | 302.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 300.25 | 302.70 | 302.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 296.45 | 301.45 | 301.53 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 306.25 | 301.25 | 300.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 307.25 | 304.31 | 302.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 304.70 | 304.71 | 303.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 304.70 | 304.71 | 303.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 305.00 | 305.33 | 304.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 305.00 | 305.33 | 304.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 305.35 | 305.38 | 304.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 304.50 | 305.38 | 304.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 304.25 | 308.74 | 307.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 304.25 | 308.74 | 307.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 304.05 | 307.80 | 307.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:00:00 | 306.10 | 307.46 | 307.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 306.45 | 307.02 | 306.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-06 15:15:00 | 336.71 | 330.32 | 324.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 328.80 | 331.77 | 331.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 325.60 | 329.17 | 330.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 321.90 | 321.80 | 325.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 321.90 | 321.80 | 325.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 319.00 | 321.19 | 324.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 318.50 | 321.19 | 324.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 318.15 | 320.41 | 323.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 316.75 | 311.00 | 315.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 15:15:00 | 302.57 | 307.23 | 311.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 15:15:00 | 302.24 | 307.23 | 311.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 307.90 | 307.37 | 310.86 | SL hit (close>ema200) qty=0.50 sl=307.37 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 252.40 | 251.27 | 251.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 260.70 | 253.17 | 252.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 259.95 | 261.40 | 259.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 259.95 | 261.40 | 259.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 259.95 | 261.40 | 259.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 259.95 | 261.40 | 259.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 258.65 | 260.85 | 259.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 258.65 | 260.85 | 259.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 260.80 | 260.84 | 259.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 259.35 | 260.84 | 259.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 258.95 | 260.46 | 259.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 258.95 | 260.46 | 259.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 258.25 | 260.02 | 259.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:45:00 | 258.55 | 260.02 | 259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 252.35 | 257.90 | 258.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 248.90 | 255.30 | 257.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 255.35 | 253.78 | 255.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 255.35 | 253.78 | 255.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 256.70 | 254.36 | 255.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 256.70 | 254.36 | 255.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 255.70 | 254.63 | 255.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 253.60 | 255.25 | 255.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 240.92 | 245.82 | 250.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 15:15:00 | 228.24 | 235.22 | 242.12 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 245.30 | 237.20 | 237.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 247.70 | 240.36 | 238.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 242.65 | 243.48 | 241.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:30:00 | 242.85 | 243.48 | 241.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 241.45 | 243.08 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 241.55 | 243.08 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 245.55 | 243.69 | 241.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:45:00 | 249.35 | 245.46 | 243.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 250.65 | 247.15 | 244.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:00:00 | 250.05 | 247.15 | 244.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 249.25 | 247.35 | 245.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 244.90 | 246.86 | 245.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 243.90 | 246.86 | 245.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 242.75 | 246.04 | 245.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 242.75 | 246.04 | 245.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 246.35 | 246.10 | 245.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 241.50 | 244.43 | 244.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 241.50 | 244.43 | 244.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 236.40 | 242.82 | 243.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 248.35 | 237.76 | 239.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 248.35 | 237.76 | 239.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 248.35 | 237.76 | 239.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 248.35 | 237.76 | 239.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 248.20 | 239.85 | 240.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 246.00 | 239.85 | 240.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 247.50 | 241.38 | 241.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 247.50 | 241.38 | 241.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 12:15:00 | 250.65 | 243.23 | 241.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 259.05 | 263.14 | 258.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 259.05 | 263.14 | 258.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 259.05 | 263.14 | 258.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 259.05 | 263.14 | 258.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 259.70 | 262.45 | 258.88 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 244.70 | 254.97 | 256.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 239.05 | 246.42 | 250.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 240.35 | 236.99 | 240.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 240.35 | 236.99 | 240.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 240.35 | 236.99 | 240.22 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 246.70 | 242.39 | 242.05 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 237.05 | 241.74 | 241.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 234.15 | 240.22 | 241.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 230.50 | 229.72 | 233.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 14:00:00 | 230.50 | 229.72 | 233.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 234.15 | 230.61 | 233.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 234.10 | 230.61 | 233.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 234.75 | 231.44 | 233.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 233.15 | 231.44 | 233.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 233.25 | 232.03 | 233.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:30:00 | 233.60 | 232.03 | 233.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 229.65 | 231.55 | 233.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:15:00 | 229.30 | 231.55 | 233.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 238.45 | 232.62 | 233.01 | SL hit (close>static) qty=1.00 sl=233.65 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 237.80 | 233.65 | 233.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 242.45 | 235.41 | 234.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 251.95 | 253.23 | 248.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:30:00 | 250.60 | 253.23 | 248.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 249.45 | 251.56 | 249.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 249.10 | 251.56 | 249.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 247.85 | 250.81 | 249.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 247.65 | 250.81 | 249.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 246.95 | 250.04 | 248.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 246.95 | 250.04 | 248.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 248.00 | 249.63 | 248.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 251.45 | 249.63 | 248.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 11:15:00 | 244.55 | 248.17 | 248.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 244.55 | 248.17 | 248.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 12:15:00 | 242.15 | 246.96 | 247.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 10:15:00 | 244.35 | 243.94 | 245.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 10:15:00 | 244.35 | 243.94 | 245.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 244.35 | 243.94 | 245.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:45:00 | 244.45 | 243.94 | 245.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 242.30 | 241.90 | 243.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:15:00 | 246.50 | 241.90 | 243.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 248.70 | 243.26 | 244.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 248.70 | 243.26 | 244.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 246.85 | 243.98 | 244.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:30:00 | 247.75 | 243.98 | 244.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 13:15:00 | 246.95 | 244.69 | 244.65 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 236.17 | 243.53 | 244.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 233.50 | 241.52 | 243.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 12:15:00 | 235.60 | 235.27 | 238.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 12:30:00 | 234.75 | 235.27 | 238.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 245.86 | 237.15 | 238.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 247.57 | 237.15 | 238.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 245.96 | 238.91 | 238.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 247.21 | 240.57 | 239.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 267.81 | 268.69 | 261.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 267.81 | 268.69 | 261.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 260.34 | 266.69 | 262.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 260.34 | 266.69 | 262.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 258.06 | 264.97 | 261.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 258.06 | 264.97 | 261.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 259.67 | 263.91 | 261.57 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 254.25 | 259.32 | 259.93 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 11:15:00 | 262.34 | 260.47 | 260.33 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 14:15:00 | 259.50 | 260.18 | 260.22 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 262.35 | 260.61 | 260.41 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 258.83 | 260.26 | 260.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 255.24 | 259.19 | 259.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 260.47 | 258.02 | 258.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 260.47 | 258.02 | 258.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 260.47 | 258.02 | 258.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 260.47 | 258.02 | 258.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 258.02 | 258.02 | 258.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 260.11 | 258.02 | 258.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 257.79 | 257.47 | 258.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:30:00 | 259.27 | 257.47 | 258.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 260.42 | 258.06 | 258.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 260.42 | 258.06 | 258.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 259.80 | 258.41 | 258.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 258.61 | 258.41 | 258.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 256.49 | 257.91 | 258.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 254.15 | 257.48 | 258.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 259.91 | 255.92 | 256.84 | SL hit (close>static) qty=1.00 sl=258.39 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 260.60 | 257.73 | 257.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 262.99 | 259.32 | 258.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 265.00 | 265.23 | 262.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 276.40 | 265.23 | 262.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 276.00 | 277.37 | 274.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 281.20 | 277.37 | 274.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 271.03 | 278.07 | 276.65 | SL hit (close<ema400) qty=1.00 sl=276.65 alert=retest1 |

### Cycle 68 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 270.09 | 275.43 | 275.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 13:15:00 | 268.08 | 273.03 | 274.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 267.48 | 267.22 | 269.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 267.48 | 267.22 | 269.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 267.48 | 267.22 | 269.50 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 272.00 | 270.33 | 270.32 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 267.90 | 269.85 | 270.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 266.65 | 268.92 | 269.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 12:15:00 | 271.00 | 269.34 | 269.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 12:15:00 | 271.00 | 269.34 | 269.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 271.00 | 269.34 | 269.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:45:00 | 271.10 | 269.34 | 269.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 270.80 | 269.63 | 269.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 270.80 | 269.63 | 269.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 271.50 | 270.00 | 269.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 272.55 | 270.88 | 270.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 13:15:00 | 270.10 | 271.01 | 270.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 13:15:00 | 270.10 | 271.01 | 270.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 270.10 | 271.01 | 270.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 270.10 | 271.01 | 270.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 271.35 | 271.08 | 270.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 272.50 | 270.97 | 270.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 272.60 | 271.01 | 270.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:15:00 | 273.30 | 271.01 | 270.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 262.45 | 271.45 | 271.37 | SL hit (close<static) qty=1.00 sl=270.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 262.10 | 269.58 | 270.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 259.50 | 266.06 | 268.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 240.40 | 239.36 | 247.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 240.40 | 239.36 | 247.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 246.05 | 242.14 | 246.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:15:00 | 245.00 | 242.14 | 246.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 245.00 | 242.72 | 246.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 237.25 | 242.72 | 246.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 248.00 | 244.59 | 244.86 | SL hit (close>static) qty=1.00 sl=247.35 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 248.75 | 245.42 | 245.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 253.05 | 247.63 | 246.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 269.50 | 269.85 | 267.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 267.25 | 269.33 | 267.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 267.25 | 269.33 | 267.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 267.25 | 269.33 | 267.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 269.90 | 269.44 | 267.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 270.70 | 269.31 | 268.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:45:00 | 270.80 | 270.18 | 268.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 262.45 | 267.91 | 268.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 262.45 | 267.91 | 268.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 258.45 | 266.01 | 267.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 263.65 | 260.74 | 263.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 263.65 | 260.74 | 263.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 263.65 | 260.74 | 263.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 263.65 | 260.74 | 263.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 262.60 | 261.11 | 263.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:00:00 | 261.30 | 262.07 | 263.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 248.23 | 252.09 | 255.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 251.96 | 251.25 | 254.85 | SL hit (close>ema200) qty=0.50 sl=251.25 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 219.00 | 214.59 | 214.31 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 11:15:00 | 212.80 | 213.99 | 214.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 212.25 | 213.64 | 213.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 15:15:00 | 213.49 | 213.30 | 213.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 216.00 | 213.30 | 213.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 216.28 | 213.89 | 213.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 216.98 | 213.89 | 213.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 217.54 | 214.62 | 214.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 219.32 | 216.55 | 215.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 216.57 | 217.03 | 215.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 216.57 | 217.03 | 215.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 216.57 | 217.03 | 215.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 216.47 | 217.03 | 215.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 215.32 | 216.52 | 215.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 215.32 | 216.52 | 215.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 215.92 | 216.40 | 215.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 216.63 | 216.19 | 215.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 216.55 | 219.32 | 219.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 215.20 | 218.50 | 218.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 215.20 | 218.50 | 218.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 214.88 | 217.78 | 218.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 212.80 | 210.93 | 212.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 212.80 | 210.93 | 212.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 212.80 | 210.93 | 212.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 213.99 | 210.93 | 212.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 212.96 | 211.34 | 212.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 213.18 | 211.34 | 212.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 214.24 | 211.92 | 212.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 214.46 | 211.92 | 212.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 213.80 | 212.29 | 212.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 213.80 | 212.29 | 212.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 215.61 | 213.63 | 213.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 216.80 | 214.52 | 213.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 214.24 | 215.20 | 214.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 214.24 | 215.20 | 214.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 214.24 | 215.20 | 214.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 213.42 | 215.20 | 214.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 215.12 | 215.19 | 214.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 215.40 | 215.19 | 214.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 215.50 | 215.21 | 214.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 215.48 | 214.95 | 214.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:30:00 | 215.33 | 215.25 | 214.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 215.42 | 215.39 | 215.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 215.42 | 215.39 | 215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 213.24 | 214.96 | 214.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 213.24 | 214.96 | 214.91 | SL hit (close<static) qty=1.00 sl=213.63 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 213.80 | 214.73 | 214.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 211.40 | 213.19 | 213.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 13:15:00 | 212.35 | 212.16 | 213.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 14:00:00 | 212.35 | 212.16 | 213.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 211.80 | 212.09 | 212.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 213.00 | 212.09 | 212.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 214.65 | 212.55 | 213.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 214.65 | 212.55 | 213.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 213.79 | 212.80 | 213.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 215.40 | 212.80 | 213.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 213.01 | 212.86 | 213.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:45:00 | 213.07 | 212.86 | 213.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 212.94 | 212.87 | 213.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 212.80 | 212.87 | 213.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 214.44 | 213.19 | 213.17 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 212.11 | 213.02 | 213.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 13:15:00 | 211.25 | 212.37 | 212.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 212.10 | 211.74 | 212.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 212.10 | 211.74 | 212.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 212.10 | 211.74 | 212.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 212.10 | 211.74 | 212.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 212.47 | 211.89 | 212.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 212.47 | 211.89 | 212.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 213.78 | 212.27 | 212.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 213.78 | 212.27 | 212.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 218.05 | 213.42 | 212.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 219.31 | 216.61 | 215.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 217.60 | 218.07 | 216.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 217.60 | 218.07 | 216.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 245.72 | 249.57 | 246.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 246.10 | 249.57 | 246.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 246.10 | 248.87 | 246.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 240.97 | 248.87 | 246.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 242.97 | 247.69 | 245.81 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 239.31 | 244.18 | 244.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 234.17 | 240.39 | 242.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 238.88 | 238.21 | 240.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 238.88 | 238.21 | 240.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 235.12 | 237.85 | 239.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 234.62 | 237.21 | 239.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 234.07 | 236.53 | 238.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 233.98 | 234.75 | 236.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 233.59 | 235.52 | 236.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 232.60 | 232.60 | 234.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 233.87 | 232.60 | 234.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 237.33 | 233.55 | 234.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 237.33 | 233.55 | 234.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 236.33 | 234.10 | 234.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 234.93 | 234.10 | 234.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 239.30 | 236.05 | 235.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 234.71 | 235.82 | 235.82 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 238.81 | 236.42 | 236.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 241.80 | 239.67 | 238.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 240.09 | 240.73 | 239.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 240.09 | 240.73 | 239.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 240.09 | 240.73 | 239.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 240.09 | 240.73 | 239.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 239.93 | 240.57 | 239.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 239.90 | 240.57 | 239.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 243.13 | 241.08 | 240.07 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 240.20 | 240.93 | 241.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 239.03 | 240.55 | 240.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 238.00 | 237.73 | 239.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 238.00 | 237.73 | 239.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 238.00 | 237.73 | 239.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 238.00 | 237.73 | 239.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 234.90 | 236.49 | 237.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:00:00 | 233.82 | 235.96 | 237.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 233.99 | 235.36 | 237.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 222.13 | 225.54 | 227.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 222.29 | 225.54 | 227.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 224.06 | 224.02 | 225.76 | SL hit (close>ema200) qty=0.50 sl=224.02 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 228.41 | 225.79 | 225.56 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 226.65 | 227.28 | 227.28 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 228.30 | 227.48 | 227.37 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 226.30 | 227.61 | 227.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 224.60 | 226.75 | 227.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 228.14 | 226.76 | 227.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 228.14 | 226.76 | 227.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 228.14 | 226.76 | 227.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 227.63 | 226.76 | 227.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 226.26 | 226.66 | 227.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 227.96 | 226.66 | 227.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 227.06 | 226.22 | 226.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 227.06 | 226.22 | 226.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 227.48 | 226.47 | 226.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 227.00 | 226.47 | 226.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 228.75 | 226.93 | 226.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 228.75 | 226.93 | 226.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 229.05 | 227.35 | 227.13 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 224.53 | 226.88 | 227.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 223.20 | 226.14 | 226.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 216.90 | 216.90 | 219.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 216.90 | 216.90 | 219.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 218.21 | 217.30 | 219.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 217.44 | 217.17 | 218.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:15:00 | 206.57 | 209.28 | 212.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 210.50 | 206.95 | 208.92 | SL hit (close>ema200) qty=0.50 sl=206.95 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 213.20 | 209.87 | 209.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 214.39 | 211.33 | 210.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 206.47 | 210.34 | 210.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 205.48 | 209.37 | 209.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 204.88 | 207.44 | 208.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 210.04 | 206.46 | 207.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 210.04 | 206.46 | 207.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 210.04 | 206.46 | 207.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 210.04 | 206.46 | 207.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 209.79 | 207.12 | 207.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 206.60 | 207.12 | 207.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 203.51 | 201.38 | 201.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 203.51 | 201.38 | 201.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 205.63 | 203.27 | 202.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 205.60 | 205.60 | 204.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 207.88 | 206.08 | 204.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 205.45 | 206.17 | 205.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 205.45 | 206.17 | 205.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 205.00 | 205.93 | 205.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 205.00 | 205.93 | 205.26 | SL hit (close<ema400) qty=1.00 sl=205.26 alert=retest1 |

### Cycle 98 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 203.11 | 204.68 | 204.80 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 207.40 | 204.74 | 204.69 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 201.57 | 204.62 | 204.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 200.21 | 202.29 | 203.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 201.30 | 200.97 | 202.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 201.30 | 200.97 | 202.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 199.91 | 200.76 | 202.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 199.91 | 200.76 | 202.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 197.60 | 199.54 | 200.94 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 204.20 | 200.09 | 199.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 206.30 | 201.33 | 200.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 208.33 | 208.59 | 206.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 206.38 | 207.77 | 206.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 206.38 | 207.77 | 206.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:15:00 | 206.01 | 207.77 | 206.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 205.87 | 207.39 | 206.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 205.60 | 207.39 | 206.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 205.53 | 207.02 | 206.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 205.55 | 207.02 | 206.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 205.55 | 206.72 | 206.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 204.50 | 206.72 | 206.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 201.95 | 205.22 | 205.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 200.77 | 204.33 | 205.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 204.72 | 202.84 | 203.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 204.72 | 202.84 | 203.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 204.72 | 202.84 | 203.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 204.72 | 202.84 | 203.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 203.35 | 202.95 | 203.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 202.99 | 203.12 | 203.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 202.00 | 203.12 | 203.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 202.31 | 203.17 | 203.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 202.84 | 202.59 | 202.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 202.45 | 202.56 | 202.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:30:00 | 202.94 | 202.56 | 202.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 202.15 | 202.48 | 202.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 202.06 | 202.48 | 202.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 202.05 | 202.19 | 202.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 201.88 | 202.28 | 202.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:00:00 | 201.88 | 202.20 | 202.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 202.52 | 202.27 | 202.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 202.52 | 202.27 | 202.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 202.84 | 202.38 | 202.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 202.84 | 202.38 | 202.48 | SL hit (close>static) qty=1.00 sl=202.79 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 208.25 | 203.56 | 203.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 210.95 | 208.07 | 206.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 208.10 | 209.39 | 207.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:00:00 | 208.10 | 209.39 | 207.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 208.90 | 209.29 | 207.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 208.05 | 209.29 | 207.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 208.83 | 208.98 | 207.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 208.83 | 208.98 | 207.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 208.93 | 210.04 | 209.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 209.00 | 210.04 | 209.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 208.43 | 209.71 | 209.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 208.43 | 209.71 | 209.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 208.69 | 209.05 | 209.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 207.58 | 208.75 | 208.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 209.77 | 208.73 | 208.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 212.47 | 209.48 | 209.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 214.50 | 210.69 | 210.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 215.70 | 216.16 | 214.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 213.70 | 216.16 | 214.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 214.61 | 215.85 | 214.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 215.00 | 215.85 | 214.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 213.74 | 215.42 | 214.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 213.30 | 215.42 | 214.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 212.95 | 214.93 | 214.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 212.95 | 214.93 | 214.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 210.59 | 213.40 | 213.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 208.11 | 211.89 | 212.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 203.59 | 203.29 | 205.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 204.25 | 203.29 | 205.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 204.58 | 203.63 | 204.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 204.58 | 203.63 | 204.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 204.60 | 204.03 | 204.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 204.52 | 204.05 | 204.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 205.29 | 204.15 | 204.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 205.29 | 204.15 | 204.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 206.48 | 204.61 | 204.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 206.48 | 204.61 | 204.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 206.69 | 205.03 | 204.83 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 204.00 | 204.73 | 204.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 202.38 | 203.94 | 204.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 203.00 | 202.31 | 202.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 201.54 | 202.16 | 202.81 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 207.51 | 203.29 | 203.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 209.34 | 204.50 | 203.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 203.00 | 205.01 | 204.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 203.00 | 205.01 | 204.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 203.00 | 205.01 | 204.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 203.00 | 205.01 | 204.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 203.85 | 204.78 | 204.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 204.59 | 204.78 | 204.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 204.07 | 204.64 | 204.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 204.07 | 204.64 | 204.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 203.52 | 204.41 | 204.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 203.52 | 204.41 | 204.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 203.64 | 204.26 | 204.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 203.87 | 204.26 | 204.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 206.12 | 204.56 | 204.32 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 202.99 | 204.13 | 204.22 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 205.50 | 204.40 | 204.29 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 201.62 | 203.84 | 204.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 200.71 | 203.21 | 203.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 11:15:00 | 199.90 | 199.70 | 201.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:00:00 | 199.90 | 199.70 | 201.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 201.51 | 200.08 | 200.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 201.51 | 200.08 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 200.90 | 200.24 | 200.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 204.51 | 200.24 | 200.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 202.57 | 200.71 | 201.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 201.32 | 200.93 | 201.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 201.34 | 201.20 | 201.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 201.00 | 201.20 | 201.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 201.20 | 201.17 | 201.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 200.93 | 201.12 | 201.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 200.52 | 200.99 | 201.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:30:00 | 200.50 | 200.71 | 200.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 191.25 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 191.27 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 190.95 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 191.14 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 190.49 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 14:15:00 | 190.47 | 198.55 | 199.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 172.76 | 168.72 | 168.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 173.28 | 171.19 | 170.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 10:15:00 | 171.81 | 171.95 | 170.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 11:00:00 | 171.81 | 171.95 | 170.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 171.17 | 171.79 | 171.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 171.17 | 171.79 | 171.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 170.60 | 171.55 | 171.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 170.60 | 171.55 | 171.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 172.29 | 171.70 | 171.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 170.74 | 171.70 | 171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 171.65 | 171.74 | 171.25 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 168.84 | 171.03 | 171.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 10:15:00 | 168.45 | 169.38 | 170.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 169.07 | 168.99 | 169.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:45:00 | 169.07 | 168.99 | 169.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 166.90 | 166.16 | 167.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 166.70 | 166.16 | 167.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 167.15 | 166.46 | 167.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 167.15 | 166.46 | 167.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 166.26 | 166.42 | 167.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 165.90 | 166.42 | 167.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 165.20 | 166.18 | 166.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:00:00 | 164.65 | 166.09 | 166.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:30:00 | 164.51 | 165.79 | 166.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 164.61 | 165.79 | 166.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:15:00 | 164.63 | 165.50 | 166.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 162.25 | 162.30 | 163.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 167.68 | 164.61 | 164.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 164.66 | 166.16 | 165.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 164.50 | 165.83 | 165.58 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 163.50 | 165.11 | 165.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 162.86 | 164.66 | 165.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 165.29 | 164.25 | 164.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 165.29 | 164.25 | 164.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 165.29 | 164.25 | 164.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 165.30 | 164.25 | 164.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 167.90 | 164.98 | 165.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 168.26 | 164.98 | 165.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 168.79 | 165.74 | 165.36 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 163.90 | 165.51 | 165.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 163.29 | 164.42 | 165.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 164.85 | 164.22 | 164.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 13:15:00 | 164.85 | 164.22 | 164.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 164.85 | 164.22 | 164.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 164.50 | 164.22 | 164.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 164.90 | 164.36 | 164.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 164.90 | 164.36 | 164.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 164.01 | 164.29 | 164.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 163.67 | 164.29 | 164.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 163.98 | 164.23 | 164.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 162.88 | 163.95 | 164.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 162.95 | 163.95 | 164.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:45:00 | 162.76 | 163.60 | 164.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 162.21 | 163.20 | 163.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 154.74 | 157.91 | 160.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 154.80 | 157.91 | 160.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 154.62 | 157.91 | 160.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 154.10 | 157.91 | 160.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 157.70 | 157.54 | 159.43 | SL hit (close>ema200) qty=0.50 sl=157.54 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 162.30 | 160.23 | 160.04 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 158.94 | 160.12 | 160.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 09:15:00 | 158.63 | 159.55 | 159.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 159.71 | 159.02 | 159.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 13:15:00 | 159.71 | 159.02 | 159.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 159.71 | 159.02 | 159.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 159.71 | 159.02 | 159.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 161.33 | 159.48 | 159.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 161.25 | 159.48 | 159.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 161.31 | 159.85 | 159.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 162.33 | 160.34 | 160.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 161.62 | 162.30 | 161.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 161.62 | 162.30 | 161.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 161.62 | 162.30 | 161.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 161.55 | 162.30 | 161.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 162.44 | 162.33 | 161.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 161.96 | 162.33 | 161.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 162.30 | 162.71 | 162.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 162.16 | 162.71 | 162.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 162.01 | 162.57 | 162.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 162.08 | 162.57 | 162.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 161.91 | 162.44 | 162.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 161.51 | 162.44 | 162.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 161.52 | 162.26 | 162.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 161.58 | 162.26 | 162.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 161.18 | 162.04 | 162.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 161.18 | 162.04 | 162.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 161.20 | 161.87 | 161.93 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 163.43 | 162.18 | 162.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 165.00 | 162.97 | 162.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 163.20 | 163.23 | 162.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 14:00:00 | 163.20 | 163.23 | 162.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 162.05 | 163.23 | 162.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 162.05 | 163.23 | 162.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 161.28 | 162.84 | 162.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 161.28 | 162.84 | 162.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 161.30 | 162.53 | 162.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 14:15:00 | 160.94 | 161.82 | 162.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 164.19 | 162.62 | 162.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 164.70 | 163.84 | 163.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 167.71 | 167.76 | 166.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 167.71 | 167.76 | 166.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 167.05 | 167.14 | 166.53 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 165.11 | 166.38 | 166.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 164.17 | 165.93 | 166.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 165.65 | 165.41 | 165.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 164.55 | 165.41 | 165.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 164.46 | 165.22 | 165.70 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 168.95 | 165.99 | 165.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 169.63 | 167.81 | 166.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 166.52 | 167.58 | 166.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 166.52 | 167.58 | 166.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 166.52 | 167.58 | 166.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 166.52 | 167.58 | 166.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 166.69 | 167.40 | 166.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 167.13 | 166.94 | 166.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 169.03 | 170.53 | 170.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 169.03 | 170.53 | 170.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 166.72 | 169.47 | 170.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 162.07 | 161.75 | 163.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 161.80 | 161.75 | 163.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 162.57 | 161.97 | 163.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 163.34 | 161.97 | 163.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 162.62 | 162.10 | 163.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 162.84 | 162.10 | 163.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 161.49 | 161.98 | 162.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 161.06 | 161.86 | 162.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 161.22 | 161.86 | 162.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 161.25 | 161.71 | 162.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 161.05 | 161.54 | 162.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 159.36 | 160.64 | 161.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:30:00 | 157.63 | 160.06 | 161.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 157.73 | 160.06 | 161.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:30:00 | 157.67 | 159.48 | 160.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.98 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 174.40 | 162.68 | 161.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 11:15:00 | 176.85 | 165.51 | 162.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 181.21 | 185.09 | 185.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 180.11 | 184.10 | 184.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 179.30 | 178.44 | 180.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 180.02 | 178.44 | 180.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 178.32 | 178.42 | 180.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:00:00 | 177.15 | 179.41 | 180.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 173.53 | 177.43 | 178.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:45:00 | 177.30 | 177.63 | 178.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 176.45 | 177.63 | 178.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 177.93 | 177.69 | 178.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 177.93 | 177.69 | 178.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 174.42 | 174.18 | 175.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 173.65 | 174.18 | 175.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 177.66 | 173.53 | 174.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 177.66 | 173.53 | 174.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 177.40 | 174.30 | 174.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 182.17 | 174.30 | 174.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 181.68 | 175.78 | 175.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 181.68 | 175.78 | 175.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 188.84 | 187.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 15:15:00 | 195.30 | 195.85 | 193.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:15:00 | 193.07 | 195.85 | 193.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 197.22 | 196.12 | 193.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 192.83 | 196.12 | 193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 193.72 | 195.34 | 193.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:45:00 | 193.61 | 195.34 | 193.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 193.43 | 194.96 | 193.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 193.43 | 194.96 | 193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 194.65 | 194.90 | 193.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:15:00 | 193.24 | 194.90 | 193.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 191.57 | 194.23 | 193.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 191.67 | 194.23 | 193.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 191.70 | 193.73 | 193.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 187.90 | 193.73 | 193.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 186.94 | 192.37 | 192.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 186.20 | 188.82 | 190.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 187.13 | 186.69 | 188.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 187.29 | 186.69 | 188.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 188.52 | 187.05 | 188.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 188.52 | 187.05 | 188.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 188.31 | 187.30 | 188.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 188.28 | 187.30 | 188.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 188.54 | 187.55 | 188.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 188.00 | 187.64 | 188.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 188.00 | 187.85 | 188.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 188.04 | 188.28 | 188.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 188.18 | 188.04 | 188.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 178.60 | 181.89 | 184.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 178.60 | 181.89 | 184.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 178.64 | 181.89 | 184.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 178.77 | 181.89 | 184.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 180.56 | 180.55 | 182.98 | SL hit (close>ema200) qty=0.50 sl=180.55 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 187.62 | 180.07 | 179.76 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 180.00 | 183.89 | 184.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 177.72 | 182.65 | 183.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 169.10 | 168.96 | 172.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 170.83 | 168.96 | 172.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 168.80 | 169.32 | 171.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 167.68 | 169.11 | 171.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 167.50 | 168.33 | 170.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 159.30 | 167.11 | 169.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 159.12 | 167.11 | 169.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 164.06 | 163.83 | 166.29 | SL hit (close>ema200) qty=0.50 sl=163.83 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 186.70 | 169.47 | 167.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 188.11 | 173.20 | 169.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 191.76 | 194.45 | 188.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 191.76 | 194.45 | 188.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 195.90 | 192.74 | 189.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 10:45:00 | 197.65 | 194.53 | 191.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 197.09 | 194.90 | 191.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 198.53 | 195.31 | 192.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 15:15:00 | 196.50 | 195.44 | 192.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 197.80 | 197.78 | 196.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 194.35 | 197.78 | 196.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 192.64 | 196.75 | 196.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 191.50 | 194.90 | 195.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 185.30 | 193.57 | 193.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 188.00 | 185.80 | 185.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 188.00 | 185.80 | 185.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 189.54 | 186.54 | 185.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 184.65 | 186.22 | 185.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 184.65 | 186.22 | 185.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 184.65 | 186.22 | 185.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 184.65 | 186.22 | 185.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 184.30 | 185.84 | 185.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 184.30 | 185.84 | 185.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 185.30 | 186.17 | 185.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 185.30 | 186.17 | 185.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 186.90 | 186.31 | 186.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 188.44 | 186.31 | 186.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 189.28 | 186.91 | 186.33 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 183.50 | 186.18 | 186.36 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 187.75 | 186.50 | 186.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 190.48 | 187.29 | 186.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 190.79 | 188.89 | 188.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 209.87 | 206.65 | 204.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 240.11 | 242.66 | 242.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 234.03 | 240.36 | 241.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 227.32 | 226.32 | 230.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:30:00 | 225.71 | 226.32 | 230.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 232.12 | 227.48 | 230.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 230.40 | 227.48 | 230.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 231.63 | 228.31 | 230.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 233.38 | 228.31 | 230.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 231.93 | 229.03 | 230.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:15:00 | 231.61 | 229.03 | 230.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 231.60 | 229.55 | 230.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 230.22 | 229.68 | 230.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 229.84 | 229.99 | 230.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 230.00 | 230.16 | 230.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 233.25 | 230.78 | 231.15 | SL hit (close>static) qty=1.00 sl=233.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 234.50 | 231.86 | 231.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 238.00 | 233.68 | 232.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 243.30 | 243.99 | 241.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 243.36 | 243.99 | 241.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:45:00 | 258.50 | 2024-05-14 09:15:00 | 269.02 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2024-05-18 09:15:00 | 278.23 | 2024-05-21 09:15:00 | 272.60 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-06-11 10:45:00 | 281.60 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-06-11 12:15:00 | 279.70 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-06-11 13:00:00 | 280.00 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-06-12 09:15:00 | 280.23 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-06-13 11:45:00 | 279.70 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-06-13 12:30:00 | 279.50 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-13 13:30:00 | 280.00 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-06-13 14:30:00 | 279.85 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-06-14 11:15:00 | 285.68 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-06-14 15:00:00 | 280.68 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-06-18 10:30:00 | 281.00 | 2024-06-19 09:15:00 | 274.18 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-06-20 15:15:00 | 282.50 | 2024-06-25 12:15:00 | 281.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-06-25 11:45:00 | 282.58 | 2024-06-25 12:15:00 | 281.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-25 12:15:00 | 282.10 | 2024-06-25 12:15:00 | 281.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-28 12:15:00 | 272.73 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-06-28 14:00:00 | 273.52 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-01 09:45:00 | 273.05 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-02 09:30:00 | 273.15 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-07-02 11:15:00 | 270.68 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2024-07-02 12:00:00 | 269.55 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-07-02 14:30:00 | 271.18 | 2024-07-03 09:15:00 | 278.40 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-07-10 09:15:00 | 285.75 | 2024-07-10 10:15:00 | 278.15 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-07-23 09:15:00 | 273.30 | 2024-07-24 10:15:00 | 278.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2024-08-02 10:15:00 | 328.58 | 2024-08-05 10:15:00 | 312.50 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest1 | 2024-08-02 11:45:00 | 324.60 | 2024-08-05 10:15:00 | 312.50 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2024-08-02 13:00:00 | 324.85 | 2024-08-05 10:15:00 | 312.50 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest1 | 2024-08-02 14:15:00 | 324.68 | 2024-08-05 10:15:00 | 312.50 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-08-09 09:15:00 | 318.77 | 2024-08-09 13:15:00 | 334.71 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-09 10:45:00 | 318.95 | 2024-08-09 13:15:00 | 334.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-09 09:15:00 | 318.77 | 2024-08-12 09:15:00 | 322.52 | STOP_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2024-08-09 10:45:00 | 318.95 | 2024-08-12 09:15:00 | 322.52 | STOP_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2024-08-21 09:15:00 | 325.23 | 2024-08-22 10:15:00 | 341.49 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 09:15:00 | 325.23 | 2024-08-23 09:15:00 | 338.80 | STOP_HIT | 0.50 | 4.17% |
| BUY | retest2 | 2024-09-17 12:45:00 | 356.10 | 2024-09-19 09:15:00 | 354.43 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-09-18 09:45:00 | 356.98 | 2024-09-19 09:15:00 | 354.43 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-18 14:45:00 | 357.25 | 2024-09-19 09:15:00 | 354.43 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-04 14:15:00 | 364.00 | 2024-10-07 14:15:00 | 345.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:45:00 | 363.40 | 2024-10-07 14:15:00 | 345.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:15:00 | 364.00 | 2024-10-08 09:15:00 | 363.30 | STOP_HIT | 0.50 | 0.19% |
| SELL | retest2 | 2024-10-07 09:45:00 | 363.40 | 2024-10-08 09:15:00 | 363.30 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2024-10-15 13:45:00 | 358.50 | 2024-10-15 14:15:00 | 366.30 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-10-18 10:30:00 | 365.20 | 2024-10-21 09:15:00 | 347.80 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2024-10-18 15:15:00 | 365.20 | 2024-10-21 09:15:00 | 347.80 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2024-11-12 12:45:00 | 307.40 | 2024-11-13 09:15:00 | 292.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 307.40 | 2024-11-14 09:15:00 | 295.80 | STOP_HIT | 0.50 | 3.77% |
| BUY | retest2 | 2024-11-29 12:00:00 | 306.10 | 2024-12-06 15:15:00 | 336.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 13:30:00 | 306.45 | 2024-12-06 15:15:00 | 337.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 318.50 | 2024-12-18 15:15:00 | 302.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 318.15 | 2024-12-18 15:15:00 | 302.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 318.50 | 2024-12-19 09:15:00 | 307.90 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2024-12-16 10:45:00 | 318.15 | 2024-12-19 09:15:00 | 307.90 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-12-18 09:30:00 | 316.75 | 2024-12-19 09:15:00 | 300.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:30:00 | 316.75 | 2024-12-19 09:15:00 | 307.90 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-01-24 09:15:00 | 253.60 | 2025-01-27 09:15:00 | 240.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 253.60 | 2025-01-27 15:15:00 | 228.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 11:45:00 | 249.35 | 2025-02-01 15:15:00 | 241.50 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-01-31 14:30:00 | 250.65 | 2025-02-01 15:15:00 | 241.50 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2025-01-31 15:00:00 | 250.05 | 2025-02-01 15:15:00 | 241.50 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-02-01 10:45:00 | 249.25 | 2025-02-01 15:15:00 | 241.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-02-04 11:15:00 | 246.00 | 2025-02-04 11:15:00 | 247.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-02-18 12:15:00 | 229.30 | 2025-02-19 09:15:00 | 238.45 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-02-25 09:15:00 | 251.45 | 2025-02-25 11:15:00 | 244.55 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-03-17 12:15:00 | 254.15 | 2025-03-18 09:15:00 | 259.91 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest1 | 2025-03-20 09:15:00 | 276.40 | 2025-03-25 09:15:00 | 271.03 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-03-24 09:15:00 | 281.20 | 2025-03-25 09:15:00 | 271.03 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-04-03 09:15:00 | 272.50 | 2025-04-04 09:15:00 | 262.45 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-04-03 10:30:00 | 272.60 | 2025-04-04 09:15:00 | 262.45 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-04-03 11:15:00 | 273.30 | 2025-04-04 09:15:00 | 262.45 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-04-09 09:15:00 | 237.25 | 2025-04-11 11:15:00 | 248.00 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-04-24 09:15:00 | 270.70 | 2025-04-25 09:15:00 | 262.45 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-04-24 10:45:00 | 270.80 | 2025-04-25 09:15:00 | 262.45 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-04-29 10:00:00 | 261.30 | 2025-04-30 14:15:00 | 248.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:00:00 | 261.30 | 2025-05-02 09:15:00 | 251.96 | STOP_HIT | 0.50 | 3.57% |
| BUY | retest2 | 2025-05-15 15:15:00 | 216.63 | 2025-05-20 13:15:00 | 215.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-20 13:00:00 | 216.55 | 2025-05-20 13:15:00 | 215.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-05-27 11:15:00 | 215.40 | 2025-05-28 14:15:00 | 213.24 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-05-27 14:00:00 | 215.50 | 2025-05-28 14:15:00 | 213.24 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-28 09:30:00 | 215.48 | 2025-05-28 14:15:00 | 213.24 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-28 10:30:00 | 215.33 | 2025-05-28 14:15:00 | 213.24 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-17 11:00:00 | 234.62 | 2025-06-20 13:15:00 | 237.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-17 11:45:00 | 234.07 | 2025-06-20 13:15:00 | 237.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-06-18 13:45:00 | 233.98 | 2025-06-20 13:15:00 | 237.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-19 10:30:00 | 233.59 | 2025-06-20 13:15:00 | 237.10 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-06-20 12:15:00 | 234.93 | 2025-06-20 13:15:00 | 237.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-03 11:00:00 | 233.82 | 2025-07-14 09:15:00 | 222.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 13:00:00 | 233.99 | 2025-07-14 09:15:00 | 222.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 11:00:00 | 233.82 | 2025-07-14 14:15:00 | 224.06 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-07-03 13:00:00 | 233.99 | 2025-07-14 14:15:00 | 224.06 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-07-30 10:30:00 | 217.44 | 2025-08-01 12:15:00 | 206.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:30:00 | 217.44 | 2025-08-04 13:15:00 | 210.50 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-08-08 09:15:00 | 206.60 | 2025-08-19 09:15:00 | 203.51 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest1 | 2025-08-21 09:30:00 | 207.88 | 2025-08-21 15:15:00 | 205.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-08 11:45:00 | 202.99 | 2025-09-11 15:15:00 | 202.84 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-09-08 15:00:00 | 202.00 | 2025-09-11 15:15:00 | 202.84 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-09 09:15:00 | 202.31 | 2025-09-11 15:15:00 | 202.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-09-10 10:30:00 | 202.84 | 2025-09-11 15:15:00 | 202.84 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-10 13:15:00 | 202.06 | 2025-09-12 09:15:00 | 208.25 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-09-11 10:00:00 | 202.05 | 2025-09-12 09:15:00 | 208.25 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-09-11 13:15:00 | 201.88 | 2025-09-12 09:15:00 | 208.25 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-11 14:00:00 | 201.88 | 2025-09-12 09:15:00 | 208.25 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-10-16 11:15:00 | 201.32 | 2025-10-17 14:15:00 | 191.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 14:30:00 | 201.34 | 2025-10-17 14:15:00 | 191.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 15:00:00 | 201.00 | 2025-10-17 14:15:00 | 190.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 10:00:00 | 201.20 | 2025-10-17 14:15:00 | 191.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 11:45:00 | 200.52 | 2025-10-17 14:15:00 | 190.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 12:30:00 | 200.50 | 2025-10-17 14:15:00 | 190.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 11:15:00 | 201.32 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-10-16 14:30:00 | 201.34 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2025-10-16 15:00:00 | 201.00 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-17 10:00:00 | 201.20 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-10-17 11:45:00 | 200.52 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-10-17 12:30:00 | 200.50 | 2025-10-21 13:15:00 | 191.16 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2025-11-21 10:00:00 | 164.65 | 2025-11-26 10:15:00 | 167.14 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-11-21 10:30:00 | 164.51 | 2025-11-26 10:15:00 | 167.14 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-21 11:00:00 | 164.61 | 2025-11-26 10:15:00 | 167.14 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-11-21 13:15:00 | 164.63 | 2025-11-26 10:15:00 | 167.14 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-05 09:45:00 | 162.88 | 2025-12-09 09:15:00 | 154.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 10:15:00 | 162.95 | 2025-12-09 09:15:00 | 154.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:45:00 | 162.76 | 2025-12-09 09:15:00 | 154.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:45:00 | 162.21 | 2025-12-09 09:15:00 | 154.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:45:00 | 162.88 | 2025-12-09 12:15:00 | 157.70 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-12-05 10:15:00 | 162.95 | 2025-12-09 12:15:00 | 157.70 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-12-05 11:45:00 | 162.76 | 2025-12-09 12:15:00 | 157.70 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2025-12-05 13:45:00 | 162.21 | 2025-12-09 12:15:00 | 157.70 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2026-01-01 14:15:00 | 167.13 | 2026-01-07 11:15:00 | 169.03 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2026-01-13 12:45:00 | 161.06 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -6.64% |
| SELL | retest2 | 2026-01-13 13:15:00 | 161.22 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2026-01-13 13:45:00 | 161.25 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest2 | 2026-01-14 12:30:00 | 161.05 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -6.64% |
| SELL | retest2 | 2026-01-16 10:30:00 | 157.63 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -8.96% |
| SELL | retest2 | 2026-01-16 11:00:00 | 157.73 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -8.89% |
| SELL | retest2 | 2026-01-16 11:30:00 | 157.67 | 2026-01-19 09:15:00 | 171.75 | STOP_HIT | 1.00 | -8.93% |
| SELL | retest2 | 2026-01-29 10:00:00 | 177.15 | 2026-02-03 09:15:00 | 181.68 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-01-30 09:15:00 | 173.53 | 2026-02-03 09:15:00 | 181.68 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2026-01-30 09:45:00 | 177.30 | 2026-02-03 09:15:00 | 181.68 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-30 10:15:00 | 176.45 | 2026-02-03 09:15:00 | 181.68 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-02-17 14:00:00 | 188.00 | 2026-02-19 14:15:00 | 178.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 15:15:00 | 188.00 | 2026-02-19 14:15:00 | 178.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 11:45:00 | 188.04 | 2026-02-19 14:15:00 | 178.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 188.18 | 2026-02-19 14:15:00 | 178.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 188.00 | 2026-02-20 11:15:00 | 180.56 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2026-02-17 15:15:00 | 188.00 | 2026-02-20 11:15:00 | 180.56 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2026-02-18 11:45:00 | 188.04 | 2026-02-20 11:15:00 | 180.56 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2026-02-18 14:30:00 | 188.18 | 2026-02-20 11:15:00 | 180.56 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2026-02-23 10:15:00 | 180.09 | 2026-02-25 09:15:00 | 184.83 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-03-06 12:15:00 | 167.68 | 2026-03-09 09:15:00 | 159.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 167.50 | 2026-03-09 09:15:00 | 159.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 167.68 | 2026-03-10 09:15:00 | 164.06 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2026-03-06 14:30:00 | 167.50 | 2026-03-10 09:15:00 | 164.06 | STOP_HIT | 0.50 | 2.05% |
| BUY | retest2 | 2026-03-16 10:45:00 | 197.65 | 2026-03-19 11:15:00 | 193.61 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-03-16 12:15:00 | 197.09 | 2026-03-19 11:15:00 | 193.61 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-03-16 14:15:00 | 198.53 | 2026-03-19 11:15:00 | 193.61 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-03-16 15:15:00 | 196.50 | 2026-03-19 11:15:00 | 193.61 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-03-23 09:15:00 | 185.30 | 2026-03-25 14:15:00 | 188.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-04-02 14:45:00 | 190.79 | 2026-04-15 09:15:00 | 209.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 15:00:00 | 230.22 | 2026-05-05 10:15:00 | 233.25 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-05-05 09:15:00 | 229.84 | 2026-05-05 10:15:00 | 233.25 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-05-05 10:15:00 | 230.00 | 2026-05-05 10:15:00 | 233.25 | STOP_HIT | 1.00 | -1.41% |
