# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 315.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 28 |
| ALERT3 | 153 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 44 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 30
- **Target hits / Stop hits / Partials:** 4 / 43 / 2
- **Avg / median % per leg:** 0.30% / -0.88%
- **Sum % (uncompounded):** 14.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 14 | 56.0% | 3 | 22 | 0 | 1.09% | 27.2% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.22% | 0.4% |
| BUY @ 3rd Alert (retest2) | 23 | 12 | 52.2% | 3 | 20 | 0 | 1.16% | 26.8% |
| SELL (all) | 24 | 5 | 20.8% | 1 | 21 | 2 | -0.52% | -12.5% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.47% | 0.5% |
| SELL @ 3rd Alert (retest2) | 23 | 4 | 17.4% | 1 | 20 | 2 | -0.56% | -12.9% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.30% | 0.9% |
| retest2 (combined) | 46 | 16 | 34.8% | 4 | 40 | 2 | 0.30% | 13.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 231.04 | 228.82 | 228.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 232.20 | 229.50 | 229.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 229.10 | 230.37 | 229.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 229.80 | 230.26 | 229.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 229.95 | 230.17 | 229.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 229.39 | 230.89 | 230.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 13:15:00 | 229.39 | 230.89 | 230.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 10:15:00 | 228.92 | 229.87 | 230.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 15:15:00 | 227.79 | 228.43 | 228.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 229.90 | 228.70 | 228.70 | SL hit (close>static) qty=1.00 sl=229.35 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 229.96 | 228.95 | 228.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 231.45 | 229.67 | 229.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 230.30 | 230.48 | 229.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:15:00 | 231.26 | 230.48 | 229.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 13:45:00 | 231.13 | 230.70 | 230.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 230.25 | 230.61 | 230.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 230.25 | 230.61 | 230.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 230.39 | 230.56 | 230.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 230.50 | 230.57 | 230.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 231.98 | 232.14 | 231.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 231.98 | 232.14 | 231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 232.00 | 232.11 | 231.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 232.00 | 232.11 | 231.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 231.71 | 232.02 | 231.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 231.71 | 232.02 | 231.75 | SL hit (close<ema400) qty=1.00 sl=231.75 alert=retest1 |

### Cycle 4 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 233.85 | 236.00 | 236.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 12:15:00 | 233.18 | 235.05 | 235.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 234.45 | 233.90 | 234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 235.19 | 234.15 | 234.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 235.19 | 234.15 | 234.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 234.71 | 234.27 | 234.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 236.14 | 234.27 | 234.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 234.71 | 234.35 | 234.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 234.35 | 234.43 | 234.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 239.00 | 235.50 | 235.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 239.00 | 235.50 | 235.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 243.54 | 237.56 | 236.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 261.98 | 263.59 | 257.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 261.75 | 263.59 | 257.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 272.90 | 270.99 | 267.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 273.81 | 270.99 | 267.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 274.29 | 271.24 | 267.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 269.61 | 273.68 | 273.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 269.61 | 273.68 | 273.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 267.62 | 270.96 | 272.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 268.30 | 264.98 | 266.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 270.35 | 266.06 | 266.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 270.35 | 266.06 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 270.80 | 267.69 | 267.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 272.63 | 269.22 | 268.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 270.01 | 270.81 | 269.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 270.02 | 270.81 | 269.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 270.09 | 270.74 | 269.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 270.78 | 270.74 | 269.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 269.61 | 270.40 | 269.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 269.61 | 270.40 | 269.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 269.69 | 270.26 | 269.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:30:00 | 269.32 | 270.26 | 269.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 269.91 | 270.19 | 269.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 269.45 | 270.19 | 269.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 269.03 | 269.96 | 269.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 269.03 | 269.96 | 269.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 269.79 | 269.92 | 269.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 271.53 | 269.92 | 269.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:45:00 | 270.75 | 272.25 | 271.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 273.30 | 275.17 | 275.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 273.30 | 275.17 | 275.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 271.95 | 274.53 | 275.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 274.30 | 273.64 | 274.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 274.45 | 273.80 | 274.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 274.45 | 273.80 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 274.70 | 273.98 | 274.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 274.70 | 273.98 | 274.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 275.15 | 274.22 | 274.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 275.15 | 274.22 | 274.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 274.40 | 274.25 | 274.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 275.15 | 274.25 | 274.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 275.30 | 274.46 | 274.51 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 276.40 | 274.85 | 274.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 11:15:00 | 278.50 | 275.58 | 275.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 276.00 | 276.19 | 275.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 275.45 | 276.19 | 275.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 275.50 | 276.05 | 275.54 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 270.70 | 274.41 | 274.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 270.15 | 271.93 | 273.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 270.50 | 270.08 | 271.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:15:00 | 268.25 | 270.08 | 271.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 261.70 | 264.48 | 266.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 261.30 | 263.94 | 265.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 261.45 | 263.94 | 265.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:00:00 | 261.55 | 263.46 | 265.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 261.30 | 263.08 | 264.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 264.00 | 263.42 | 264.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 264.15 | 263.42 | 264.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 267.00 | 263.94 | 264.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 267.00 | 263.94 | 264.20 | SL hit (close>ema400) qty=1.00 sl=264.20 alert=retest1 |

### Cycle 11 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 267.30 | 264.61 | 264.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 269.40 | 266.37 | 265.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 270.20 | 270.57 | 269.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 273.40 | 270.57 | 269.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 274.50 | 271.36 | 269.97 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 269.75 | 272.26 | 272.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 266.70 | 271.15 | 271.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 252.65 | 250.71 | 252.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 252.65 | 250.71 | 252.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 255.20 | 251.60 | 253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 254.90 | 251.60 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 255.95 | 252.47 | 253.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 256.80 | 252.47 | 253.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 260.35 | 254.05 | 254.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 263.30 | 255.90 | 254.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 258.45 | 258.67 | 256.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 256.85 | 258.21 | 256.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:30:00 | 256.60 | 258.21 | 256.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 257.05 | 257.98 | 256.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 258.40 | 257.98 | 256.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 257.75 | 257.93 | 256.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 257.10 | 257.93 | 256.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 257.00 | 257.99 | 257.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 257.00 | 257.99 | 257.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 257.80 | 257.95 | 257.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 256.10 | 257.95 | 257.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 256.15 | 257.59 | 257.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 255.20 | 257.59 | 257.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 256.00 | 257.28 | 257.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 255.95 | 257.28 | 257.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 256.05 | 256.96 | 256.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 256.05 | 256.96 | 256.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 256.00 | 256.77 | 256.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 255.40 | 256.40 | 256.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 256.85 | 256.49 | 256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 254.50 | 256.09 | 256.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 254.10 | 256.09 | 256.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 259.55 | 256.79 | 256.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 259.55 | 256.79 | 256.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 261.90 | 257.81 | 257.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 258.10 | 260.40 | 259.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 257.35 | 259.79 | 258.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 259.40 | 259.79 | 258.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 256.40 | 259.11 | 258.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 255.30 | 259.11 | 258.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 264.15 | 260.12 | 259.19 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 255.50 | 258.61 | 258.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 253.90 | 256.33 | 257.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 256.05 | 255.93 | 257.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 256.05 | 255.93 | 257.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 256.05 | 255.33 | 256.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:30:00 | 256.55 | 255.33 | 256.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 256.25 | 255.51 | 256.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:30:00 | 256.50 | 255.51 | 256.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 265.35 | 257.33 | 257.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 268.20 | 263.85 | 260.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 265.70 | 266.18 | 263.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 265.00 | 266.18 | 263.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 265.70 | 267.87 | 266.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 265.70 | 267.87 | 266.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 264.95 | 267.29 | 266.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 265.30 | 267.29 | 266.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 265.40 | 266.91 | 265.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 264.80 | 266.91 | 265.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 265.55 | 266.42 | 265.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:30:00 | 265.60 | 266.42 | 265.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 265.90 | 266.32 | 265.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 267.15 | 266.55 | 266.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 267.20 | 266.49 | 266.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 264.85 | 265.97 | 265.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 264.85 | 265.97 | 265.98 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 267.35 | 266.16 | 266.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 272.00 | 267.78 | 266.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 13:15:00 | 269.75 | 269.94 | 268.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 269.75 | 269.94 | 268.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 269.00 | 269.58 | 268.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 267.80 | 269.58 | 268.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 267.70 | 269.20 | 268.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 267.05 | 269.20 | 268.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 267.95 | 268.95 | 268.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 267.35 | 268.95 | 268.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 269.10 | 268.98 | 268.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 268.65 | 268.98 | 268.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 270.95 | 269.38 | 268.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:15:00 | 272.50 | 269.38 | 268.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 266.95 | 268.89 | 268.58 | SL hit (close<static) qty=1.00 sl=268.50 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 265.85 | 268.29 | 268.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 263.75 | 267.10 | 267.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 268.10 | 264.21 | 265.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 269.70 | 265.31 | 265.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 269.70 | 265.31 | 265.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 271.90 | 266.63 | 266.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 272.55 | 268.64 | 267.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 285.50 | 286.19 | 281.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 285.50 | 286.19 | 281.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 285.00 | 284.87 | 283.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 285.35 | 284.45 | 283.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 285.45 | 284.56 | 283.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 285.40 | 284.58 | 283.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 288.10 | 289.45 | 289.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 288.10 | 289.45 | 289.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 286.80 | 288.64 | 289.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 288.10 | 287.79 | 288.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 288.50 | 287.94 | 288.50 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 291.15 | 288.92 | 288.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 15:15:00 | 291.95 | 290.96 | 290.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 292.70 | 294.27 | 292.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 293.10 | 294.04 | 292.95 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 289.35 | 292.56 | 292.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 286.60 | 290.33 | 291.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 289.40 | 288.43 | 289.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 292.25 | 289.19 | 290.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 292.25 | 289.19 | 290.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 292.80 | 289.91 | 290.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 292.80 | 289.91 | 290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 293.10 | 290.55 | 290.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 294.35 | 292.10 | 291.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 291.65 | 292.63 | 292.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 290.65 | 292.23 | 291.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 290.65 | 292.23 | 291.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 290.40 | 291.87 | 291.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 289.65 | 291.87 | 291.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 289.05 | 291.30 | 291.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 286.90 | 290.42 | 291.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 279.50 | 278.80 | 281.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 280.40 | 278.80 | 281.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 282.20 | 279.19 | 280.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 282.20 | 279.19 | 280.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 282.65 | 279.88 | 281.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 282.65 | 279.88 | 281.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 281.25 | 281.12 | 281.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:30:00 | 281.55 | 281.12 | 281.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 280.05 | 280.91 | 281.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 279.30 | 280.34 | 280.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 282.30 | 280.74 | 281.06 | SL hit (close>static) qty=1.00 sl=281.75 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 283.30 | 281.33 | 281.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 285.65 | 282.27 | 281.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 282.85 | 283.50 | 282.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 282.25 | 283.25 | 282.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 282.10 | 283.25 | 282.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 282.75 | 283.15 | 282.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:45:00 | 283.05 | 283.08 | 282.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 284.40 | 283.68 | 282.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 284.65 | 290.78 | 291.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 284.65 | 290.78 | 291.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 283.65 | 285.62 | 286.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 287.75 | 285.26 | 286.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 288.15 | 285.84 | 286.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 288.25 | 285.84 | 286.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 288.20 | 286.31 | 286.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 288.20 | 286.31 | 286.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 288.80 | 286.81 | 286.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 289.40 | 287.58 | 287.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 287.70 | 287.74 | 287.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:45:00 | 287.65 | 287.74 | 287.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 287.50 | 287.69 | 287.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 287.90 | 287.69 | 287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 286.75 | 287.75 | 287.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 286.75 | 287.75 | 287.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 287.00 | 287.60 | 287.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 289.90 | 287.60 | 287.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 288.35 | 287.75 | 287.52 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 286.80 | 287.33 | 287.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 284.95 | 286.86 | 287.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 286.45 | 285.51 | 286.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 286.50 | 285.71 | 286.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 286.50 | 285.71 | 286.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 287.90 | 286.14 | 286.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 287.90 | 286.14 | 286.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 287.95 | 286.51 | 286.44 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 280.05 | 285.20 | 285.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 277.70 | 279.96 | 281.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 279.25 | 276.87 | 278.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 276.70 | 276.84 | 278.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 275.50 | 276.84 | 278.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 275.20 | 276.95 | 278.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 275.30 | 276.19 | 277.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 275.90 | 276.12 | 277.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 277.45 | 276.39 | 277.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 278.25 | 276.39 | 277.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 279.50 | 277.01 | 277.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 279.50 | 277.01 | 277.31 | SL hit (close>static) qty=1.00 sl=279.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 280.00 | 277.61 | 277.55 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 276.40 | 277.37 | 277.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 273.75 | 276.40 | 276.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 278.85 | 276.30 | 276.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 274.15 | 275.87 | 276.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 273.50 | 275.19 | 276.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 275.20 | 271.70 | 271.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 275.20 | 271.70 | 271.38 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 270.20 | 271.57 | 271.69 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 274.05 | 272.12 | 271.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 274.90 | 272.67 | 272.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 275.00 | 275.87 | 274.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 274.20 | 275.54 | 274.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 274.20 | 275.54 | 274.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 275.30 | 275.49 | 274.56 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 11:15:00 | 272.85 | 274.09 | 274.14 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 275.05 | 274.28 | 274.20 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 273.85 | 274.26 | 274.27 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 284.20 | 276.25 | 275.17 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 276.45 | 279.62 | 279.91 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 280.85 | 279.99 | 279.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 282.05 | 280.51 | 280.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 278.55 | 281.13 | 280.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 278.70 | 280.65 | 280.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 278.45 | 280.65 | 280.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 279.20 | 280.36 | 280.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 277.05 | 279.40 | 279.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 279.65 | 279.01 | 279.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 278.65 | 278.94 | 279.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 279.80 | 278.94 | 279.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 278.45 | 278.84 | 279.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 277.35 | 278.82 | 279.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:30:00 | 277.00 | 278.42 | 279.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 277.30 | 277.12 | 278.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:30:00 | 277.45 | 277.31 | 278.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 277.90 | 277.43 | 278.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:30:00 | 277.15 | 277.48 | 278.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 279.40 | 278.02 | 278.32 | SL hit (close>static) qty=1.00 sl=278.80 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 282.40 | 279.05 | 278.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 284.50 | 280.77 | 279.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 284.60 | 285.22 | 282.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 284.70 | 285.22 | 282.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 283.00 | 284.92 | 284.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 283.15 | 284.92 | 284.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 285.00 | 284.94 | 284.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 285.55 | 284.86 | 284.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 285.60 | 284.80 | 284.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 282.40 | 283.91 | 284.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 282.40 | 283.91 | 284.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 278.90 | 282.48 | 283.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 279.50 | 278.85 | 280.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 277.20 | 278.85 | 280.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 276.10 | 274.96 | 276.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 276.40 | 274.96 | 276.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 277.30 | 275.42 | 276.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 277.40 | 275.42 | 276.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 277.95 | 275.93 | 276.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 277.95 | 275.93 | 276.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 279.60 | 276.66 | 276.64 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 273.85 | 276.15 | 276.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 272.10 | 275.34 | 276.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 274.55 | 273.22 | 274.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 275.10 | 273.60 | 274.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 275.10 | 273.60 | 274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 276.20 | 274.12 | 274.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 276.20 | 274.12 | 274.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 274.25 | 274.71 | 274.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 282.30 | 274.71 | 274.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 281.80 | 276.13 | 275.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 285.85 | 282.30 | 280.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 285.80 | 286.07 | 284.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:45:00 | 285.75 | 286.07 | 284.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 282.75 | 285.31 | 284.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 283.30 | 285.31 | 284.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 283.90 | 285.03 | 284.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 283.90 | 285.03 | 284.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 282.70 | 283.71 | 283.83 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 285.00 | 284.00 | 283.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 288.60 | 285.97 | 285.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 285.80 | 287.84 | 286.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 287.00 | 287.68 | 286.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 286.70 | 287.68 | 286.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 284.80 | 287.10 | 286.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 285.35 | 287.10 | 286.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 285.85 | 286.85 | 286.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 287.30 | 286.85 | 286.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-24 15:15:00 | 316.03 | 309.71 | 302.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 306.35 | 309.38 | 309.51 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 311.00 | 309.54 | 309.53 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 306.60 | 309.21 | 309.41 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 313.65 | 309.23 | 309.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 314.50 | 312.25 | 310.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 313.55 | 313.70 | 312.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:00:00 | 313.55 | 313.70 | 312.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 310.15 | 312.99 | 311.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 310.15 | 312.99 | 311.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 311.10 | 312.61 | 311.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 310.15 | 312.61 | 311.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 307.00 | 311.23 | 311.33 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 312.80 | 309.15 | 309.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 319.90 | 312.57 | 310.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 313.00 | 313.92 | 311.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:45:00 | 314.75 | 313.92 | 311.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 307.95 | 312.73 | 311.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 307.95 | 312.73 | 311.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 310.75 | 312.33 | 311.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:45:00 | 311.55 | 312.43 | 311.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 311.35 | 311.39 | 311.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 308.50 | 310.81 | 310.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 308.50 | 310.81 | 310.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 307.40 | 309.52 | 310.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 303.50 | 297.15 | 300.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 304.65 | 298.65 | 300.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 304.65 | 298.65 | 300.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 308.20 | 302.62 | 302.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 309.40 | 306.53 | 304.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 11:15:00 | 312.75 | 312.82 | 310.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 11:45:00 | 312.15 | 312.82 | 310.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 307.30 | 311.96 | 310.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 306.40 | 311.96 | 310.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 305.55 | 310.67 | 310.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 307.75 | 310.67 | 310.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 305.20 | 309.58 | 309.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 303.25 | 308.31 | 309.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 300.90 | 300.29 | 303.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 302.75 | 300.29 | 303.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 295.10 | 299.25 | 302.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 300.60 | 299.25 | 302.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 299.95 | 298.45 | 300.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:30:00 | 300.35 | 298.45 | 300.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 299.45 | 298.65 | 300.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 302.40 | 298.65 | 300.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 299.90 | 298.90 | 300.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 298.25 | 298.78 | 300.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 283.34 | 293.92 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 09:15:00 | 268.43 | 286.33 | 289.87 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 289.40 | 280.13 | 279.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 292.75 | 285.32 | 282.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 293.25 | 291.71 | 289.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 292.45 | 291.58 | 290.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 298.35 | 304.62 | 305.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 298.35 | 304.62 | 305.11 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 306.65 | 303.29 | 303.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 311.10 | 307.71 | 305.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 307.20 | 308.21 | 306.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:45:00 | 307.60 | 308.21 | 306.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 308.05 | 308.18 | 306.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 306.55 | 308.18 | 306.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 306.55 | 307.83 | 306.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 305.90 | 307.83 | 306.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 306.95 | 307.66 | 306.86 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 305.25 | 306.39 | 306.41 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 307.15 | 306.48 | 306.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 308.50 | 306.93 | 306.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 305.95 | 306.85 | 306.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 305.00 | 306.48 | 306.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 304.35 | 306.05 | 306.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 304.50 | 304.36 | 305.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 304.50 | 304.36 | 305.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 304.50 | 304.39 | 305.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 304.50 | 304.39 | 305.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 305.70 | 304.65 | 305.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 303.40 | 304.50 | 305.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:15:00 | 288.23 | 298.56 | 301.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 293.75 | 293.16 | 296.85 | SL hit (close>ema200) qty=0.50 sl=293.16 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 258.55 | 255.22 | 255.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 262.65 | 257.27 | 256.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 265.00 | 266.29 | 263.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 265.00 | 266.29 | 263.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 264.20 | 265.87 | 263.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 264.20 | 265.87 | 263.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 262.35 | 264.93 | 263.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 262.35 | 264.93 | 263.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 263.20 | 264.58 | 263.36 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 259.00 | 262.28 | 262.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 250.55 | 259.45 | 261.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 252.75 | 252.35 | 256.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 15:00:00 | 252.75 | 252.35 | 256.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 250.65 | 252.11 | 255.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 249.35 | 251.56 | 255.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 266.20 | 255.06 | 255.20 | SL hit (close>static) qty=1.00 sl=257.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 269.15 | 257.88 | 256.47 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 252.75 | 257.28 | 257.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 251.65 | 254.31 | 255.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 259.70 | 256.03 | 255.54 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 251.80 | 254.98 | 255.24 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 257.75 | 255.50 | 255.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 258.90 | 256.18 | 255.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 256.15 | 258.06 | 257.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 255.00 | 257.45 | 256.89 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 254.75 | 256.53 | 256.54 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 264.55 | 257.80 | 257.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 269.50 | 260.14 | 258.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 265.25 | 266.00 | 262.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 264.20 | 266.00 | 262.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 262.60 | 264.94 | 263.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 263.45 | 264.94 | 263.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 263.45 | 264.64 | 263.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 263.80 | 264.64 | 263.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 263.55 | 264.42 | 263.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 266.10 | 264.42 | 263.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 265.10 | 265.92 | 264.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 11:15:00 | 292.71 | 284.29 | 278.46 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 14:15:00 | 229.95 | 2025-05-15 13:15:00 | 229.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-05-21 15:15:00 | 227.79 | 2025-05-22 12:15:00 | 229.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2025-05-26 09:15:00 | 231.26 | 2025-05-29 13:15:00 | 231.71 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest1 | 2025-05-26 13:45:00 | 231.13 | 2025-05-29 13:15:00 | 231.71 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-05-30 11:30:00 | 235.50 | 2025-06-04 10:15:00 | 233.85 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-05 15:15:00 | 234.35 | 2025-06-06 10:15:00 | 239.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-13 10:15:00 | 273.81 | 2025-06-17 12:15:00 | 269.61 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-13 11:15:00 | 274.29 | 2025-06-17 12:15:00 | 269.61 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-06-25 09:15:00 | 271.53 | 2025-07-02 13:15:00 | 273.30 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-06-27 09:45:00 | 270.75 | 2025-07-02 13:15:00 | 273.30 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest1 | 2025-07-09 09:15:00 | 268.25 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-07-11 10:30:00 | 261.30 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-07-11 11:15:00 | 261.45 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-07-11 12:00:00 | 261.55 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-14 09:15:00 | 261.30 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-07 13:15:00 | 254.10 | 2025-08-07 14:15:00 | 259.55 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-21 11:00:00 | 267.15 | 2025-08-21 15:15:00 | 264.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-21 11:30:00 | 267.20 | 2025-08-21 15:15:00 | 264.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-26 13:15:00 | 272.50 | 2025-08-26 14:15:00 | 266.95 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-09-05 13:15:00 | 285.35 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-05 14:15:00 | 285.45 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-09-05 15:15:00 | 285.40 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-09-30 13:30:00 | 279.30 | 2025-09-30 15:15:00 | 282.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-03 13:45:00 | 283.05 | 2025-10-10 10:15:00 | 284.65 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-10-03 14:30:00 | 284.40 | 2025-10-10 10:15:00 | 284.65 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-10-28 14:15:00 | 275.50 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-29 09:15:00 | 275.20 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-29 12:30:00 | 275.30 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-29 15:00:00 | 275.90 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-31 13:45:00 | 273.50 | 2025-11-06 12:15:00 | 275.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-24 12:30:00 | 277.35 | 2025-11-25 14:15:00 | 279.40 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-24 13:30:00 | 277.00 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-25 10:00:00 | 277.30 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-25 10:30:00 | 277.45 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-25 12:30:00 | 277.15 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-11-28 15:15:00 | 285.55 | 2025-12-01 13:15:00 | 282.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-01 10:15:00 | 285.60 | 2025-12-01 13:15:00 | 282.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-19 11:15:00 | 287.30 | 2025-12-24 15:15:00 | 316.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-08 12:45:00 | 311.55 | 2026-01-08 15:15:00 | 308.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-01-08 14:45:00 | 311.35 | 2026-01-08 15:15:00 | 308.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-23 10:45:00 | 298.25 | 2026-01-30 09:15:00 | 283.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 298.25 | 2026-02-01 09:15:00 | 268.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-05 14:30:00 | 293.25 | 2026-02-13 10:15:00 | 298.35 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2026-02-06 11:30:00 | 292.45 | 2026-02-13 10:15:00 | 298.35 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2026-02-25 09:30:00 | 303.40 | 2026-02-26 09:15:00 | 288.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 303.40 | 2026-02-26 15:15:00 | 293.75 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2026-03-24 11:00:00 | 249.35 | 2026-03-25 09:15:00 | 266.20 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2026-04-10 09:15:00 | 266.10 | 2026-04-22 11:15:00 | 292.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 09:45:00 | 265.10 | 2026-04-22 11:15:00 | 291.61 | TARGET_HIT | 1.00 | 10.00% |
