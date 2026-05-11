# Schneider Electric Infrastructure Ltd. (SCHNEIDER)

## Backtest Summary

- **Window:** 2023-05-26 09:15:00 → 2026-05-11 15:15:00 (5066 bars)
- **Last close:** 1316.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 189 |
| ALERT1 | 142 |
| ALERT2 | 141 |
| ALERT2_SKIP | 79 |
| ALERT3 | 1049 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 15:15:00 | 247.95 | 251.72 | 251.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 11:15:00 | 244.00 | 248.89 | 250.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 09:15:00 | 247.65 | 246.94 | 248.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 247.65 | 246.94 | 248.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 247.65 | 246.94 | 248.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 09:15:00 | 247.65 | 246.94 | 248.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 248.60 | 247.27 | 248.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 10:15:00 | 248.60 | 247.27 | 248.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 246.50 | 247.12 | 248.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:15:00 | 246.50 | 247.12 | 248.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 250.05 | 247.53 | 248.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:15:00 | 250.05 | 247.53 | 248.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 245.30 | 247.09 | 248.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:15:00 | 245.30 | 247.09 | 248.14 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 247.30 | 246.85 | 247.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 247.30 | 246.85 | 247.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 243.15 | 246.11 | 247.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 10:15:00 | 243.15 | 246.11 | 247.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 246.35 | 243.98 | 245.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:15:00 | 246.35 | 243.98 | 245.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 246.40 | 244.47 | 245.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:15:00 | 246.40 | 244.47 | 245.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 249.25 | 245.76 | 245.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:15:00 | 249.25 | 245.76 | 245.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 249.05 | 246.42 | 246.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 250.15 | 247.74 | 246.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 09:15:00 | 245.95 | 247.38 | 246.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 245.95 | 247.38 | 246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 245.95 | 247.38 | 246.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:15:00 | 245.95 | 247.38 | 246.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 245.55 | 247.01 | 246.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:15:00 | 245.55 | 247.01 | 246.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 248.40 | 247.56 | 247.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 15:15:00 | 248.40 | 247.56 | 247.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 243.10 | 246.67 | 246.74 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 15:15:00 | 247.25 | 246.76 | 246.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 249.70 | 247.35 | 246.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 13:15:00 | 247.75 | 247.82 | 247.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 13:15:00 | 247.75 | 247.82 | 247.37 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 245.70 | 247.40 | 247.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:15:00 | 245.70 | 247.40 | 247.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 246.45 | 247.21 | 247.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 15:15:00 | 246.45 | 247.21 | 247.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 09:15:00 | 245.60 | 246.89 | 247.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 10:15:00 | 244.45 | 246.40 | 246.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 14:15:00 | 245.60 | 245.48 | 246.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 245.60 | 245.48 | 246.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 245.60 | 245.48 | 246.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:15:00 | 245.60 | 245.48 | 246.14 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 244.35 | 245.25 | 245.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 244.35 | 245.25 | 245.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 247.00 | 243.09 | 243.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 247.00 | 243.09 | 243.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 10:15:00 | 257.80 | 246.03 | 244.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 12:15:00 | 263.85 | 252.07 | 247.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 257.95 | 258.64 | 254.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:15:00 | 257.95 | 258.64 | 254.14 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 252.60 | 257.00 | 254.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 252.60 | 257.00 | 254.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 251.45 | 255.89 | 254.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:15:00 | 251.45 | 255.89 | 254.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 252.50 | 255.21 | 254.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 11:15:00 | 252.50 | 255.21 | 254.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 13:15:00 | 249.30 | 253.61 | 253.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 248.75 | 252.64 | 253.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 253.20 | 250.92 | 251.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 13:15:00 | 253.20 | 250.92 | 251.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 253.20 | 250.92 | 251.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:15:00 | 253.20 | 250.92 | 251.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 258.40 | 252.41 | 252.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:15:00 | 258.40 | 252.41 | 252.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 266.10 | 255.15 | 253.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 12:15:00 | 266.95 | 264.09 | 261.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 13:15:00 | 264.05 | 264.08 | 262.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 13:15:00 | 264.05 | 264.08 | 262.17 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 274.85 | 276.76 | 272.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:15:00 | 274.85 | 276.76 | 272.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 275.25 | 275.74 | 273.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:15:00 | 275.25 | 275.74 | 273.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 271.70 | 274.93 | 273.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:15:00 | 271.70 | 274.93 | 273.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 276.65 | 275.27 | 273.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:15:00 | 276.65 | 275.27 | 273.91 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 277.00 | 277.30 | 275.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 277.00 | 277.30 | 275.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 274.20 | 276.68 | 275.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:15:00 | 274.20 | 276.68 | 275.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 273.10 | 275.96 | 275.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:15:00 | 273.10 | 275.96 | 275.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 13:15:00 | 276.45 | 276.00 | 275.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 13:15:00 | 276.45 | 276.00 | 275.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 275.45 | 275.89 | 275.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:15:00 | 275.45 | 275.89 | 275.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 275.55 | 275.82 | 275.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 15:15:00 | 275.55 | 275.82 | 275.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 277.30 | 276.12 | 275.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:15:00 | 277.30 | 276.12 | 275.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 282.90 | 277.48 | 276.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:15:00 | 282.90 | 277.48 | 276.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 279.40 | 281.20 | 279.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:15:00 | 279.40 | 281.20 | 279.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 280.20 | 281.00 | 279.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 13:15:00 | 280.20 | 281.00 | 279.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 279.80 | 280.76 | 279.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:15:00 | 279.80 | 280.76 | 279.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 279.50 | 280.51 | 279.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 15:15:00 | 279.50 | 280.51 | 279.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 279.45 | 280.45 | 279.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:15:00 | 279.45 | 280.45 | 279.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 278.15 | 279.99 | 279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 11:15:00 | 278.15 | 279.99 | 279.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 278.85 | 279.76 | 279.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:15:00 | 278.85 | 279.76 | 279.56 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 273.00 | 278.41 | 278.96 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 280.70 | 278.42 | 278.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 303.25 | 284.12 | 281.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 10:15:00 | 291.00 | 295.68 | 292.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 10:15:00 | 291.00 | 295.68 | 292.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 291.00 | 295.68 | 292.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:15:00 | 291.00 | 295.68 | 292.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 290.15 | 294.58 | 292.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:15:00 | 290.15 | 294.58 | 292.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 287.00 | 290.32 | 290.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 11:15:00 | 284.90 | 288.60 | 289.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 286.65 | 286.07 | 287.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 286.65 | 286.07 | 287.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 286.65 | 286.07 | 287.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 286.65 | 286.07 | 287.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 283.75 | 284.09 | 285.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 283.75 | 284.09 | 285.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 284.00 | 284.08 | 285.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:15:00 | 284.00 | 284.08 | 285.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 285.70 | 283.90 | 284.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:15:00 | 285.70 | 283.90 | 284.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 284.75 | 284.07 | 284.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:15:00 | 284.75 | 284.07 | 284.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 286.65 | 284.58 | 285.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:15:00 | 286.65 | 284.58 | 285.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 286.85 | 285.04 | 285.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:15:00 | 286.85 | 285.04 | 285.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 284.70 | 284.23 | 284.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:15:00 | 284.70 | 284.23 | 284.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 283.25 | 284.04 | 284.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:15:00 | 283.25 | 284.04 | 284.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 281.45 | 279.43 | 280.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:15:00 | 281.45 | 279.43 | 280.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 282.60 | 280.06 | 280.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:15:00 | 282.60 | 280.06 | 280.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 284.25 | 280.90 | 281.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:15:00 | 284.25 | 280.90 | 281.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 14:15:00 | 283.75 | 281.73 | 281.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 15:15:00 | 284.00 | 282.18 | 281.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 283.20 | 284.06 | 283.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 12:15:00 | 283.20 | 284.06 | 283.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 283.20 | 284.06 | 283.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:15:00 | 283.20 | 284.06 | 283.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 284.50 | 284.15 | 283.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:15:00 | 284.50 | 284.15 | 283.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 283.55 | 284.03 | 283.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:15:00 | 283.55 | 284.03 | 283.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 282.30 | 283.86 | 283.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:15:00 | 282.30 | 283.86 | 283.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 280.20 | 283.13 | 283.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:15:00 | 280.20 | 283.13 | 283.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 280.20 | 282.54 | 282.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 275.85 | 281.20 | 282.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 281.30 | 280.83 | 281.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 281.30 | 280.83 | 281.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 281.30 | 280.83 | 281.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:15:00 | 281.30 | 280.83 | 281.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 284.95 | 281.65 | 282.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:15:00 | 284.95 | 281.65 | 282.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 284.05 | 282.13 | 282.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:15:00 | 284.05 | 282.13 | 282.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 12:15:00 | 283.00 | 282.31 | 282.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 14:15:00 | 286.00 | 283.16 | 282.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 10:15:00 | 288.50 | 289.54 | 287.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 10:15:00 | 288.50 | 289.54 | 287.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 288.50 | 289.54 | 287.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:15:00 | 288.50 | 289.54 | 287.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 287.85 | 289.65 | 288.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:15:00 | 287.85 | 289.65 | 288.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 284.95 | 288.71 | 287.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:15:00 | 284.95 | 288.71 | 287.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 283.25 | 287.62 | 287.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 15:15:00 | 283.25 | 287.62 | 287.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 289.75 | 288.92 | 288.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:15:00 | 289.75 | 288.92 | 288.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 289.25 | 288.98 | 288.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 12:15:00 | 289.25 | 288.98 | 288.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 290.45 | 289.77 | 288.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 290.45 | 289.77 | 288.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 286.85 | 289.18 | 288.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:15:00 | 286.85 | 289.18 | 288.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 288.75 | 289.10 | 288.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:15:00 | 288.75 | 289.10 | 288.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 288.95 | 289.07 | 288.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 12:15:00 | 288.95 | 289.07 | 288.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 289.05 | 289.06 | 288.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 13:15:00 | 289.05 | 289.06 | 288.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 286.95 | 288.64 | 288.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:15:00 | 286.95 | 288.64 | 288.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 290.75 | 289.06 | 288.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:15:00 | 290.75 | 289.06 | 288.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 331.50 | 335.64 | 330.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:15:00 | 331.50 | 335.64 | 330.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 333.50 | 335.21 | 331.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:15:00 | 333.50 | 335.21 | 331.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 331.05 | 334.38 | 331.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:15:00 | 331.05 | 334.38 | 331.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 328.50 | 333.20 | 330.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:15:00 | 328.50 | 333.20 | 330.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 330.55 | 332.67 | 330.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 13:15:00 | 330.55 | 332.67 | 330.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 331.15 | 332.37 | 330.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:15:00 | 331.15 | 332.37 | 330.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 331.05 | 332.10 | 330.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:15:00 | 331.05 | 332.10 | 330.88 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 329.15 | 331.51 | 330.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 329.15 | 331.51 | 330.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 330.45 | 331.30 | 330.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:15:00 | 330.45 | 331.30 | 330.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 329.30 | 330.90 | 330.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:15:00 | 329.30 | 330.90 | 330.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 328.80 | 330.48 | 330.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:15:00 | 328.80 | 330.48 | 330.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 13:15:00 | 328.75 | 330.13 | 330.26 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 331.70 | 330.36 | 330.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 345.65 | 333.42 | 331.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 339.15 | 340.75 | 338.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 12:15:00 | 339.15 | 340.75 | 338.08 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 337.60 | 340.12 | 338.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:15:00 | 337.60 | 340.12 | 338.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 336.65 | 339.42 | 337.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:15:00 | 336.65 | 339.42 | 337.91 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 337.65 | 339.07 | 337.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:15:00 | 337.65 | 339.07 | 337.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 339.40 | 339.14 | 338.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:15:00 | 339.40 | 339.14 | 338.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 338.90 | 339.09 | 338.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 10:15:00 | 338.90 | 339.09 | 338.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 337.65 | 338.80 | 338.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:15:00 | 337.65 | 338.80 | 338.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 336.80 | 338.40 | 337.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 12:15:00 | 336.80 | 338.40 | 337.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 337.00 | 338.12 | 337.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:15:00 | 337.00 | 338.12 | 337.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 336.90 | 337.88 | 337.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:15:00 | 336.90 | 337.88 | 337.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 335.00 | 337.30 | 337.52 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-25 15:15:00 | 339.50 | 337.21 | 337.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 09:15:00 | 344.25 | 338.62 | 337.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 13:15:00 | 344.65 | 344.77 | 342.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-29 13:15:00 | 344.65 | 344.77 | 342.62 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 343.55 | 344.93 | 343.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:15:00 | 343.55 | 344.93 | 343.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 343.90 | 344.72 | 343.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 11:15:00 | 343.90 | 344.72 | 343.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 342.50 | 344.28 | 343.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:15:00 | 342.50 | 344.28 | 343.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 341.75 | 343.77 | 343.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:15:00 | 341.75 | 343.77 | 343.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 341.10 | 343.24 | 343.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:15:00 | 341.10 | 343.24 | 343.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 15:15:00 | 341.50 | 342.89 | 342.91 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 355.45 | 345.40 | 344.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 11:15:00 | 360.75 | 350.13 | 346.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 09:15:00 | 352.40 | 353.42 | 349.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 352.40 | 353.42 | 349.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 352.40 | 353.42 | 349.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 352.40 | 353.42 | 349.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 351.70 | 353.41 | 351.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 351.70 | 353.41 | 351.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 351.25 | 352.98 | 351.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 10:15:00 | 351.25 | 352.98 | 351.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 353.25 | 353.03 | 351.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:15:00 | 353.25 | 353.03 | 351.65 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 358.90 | 359.54 | 357.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:15:00 | 358.90 | 359.54 | 357.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 357.65 | 359.54 | 358.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 11:15:00 | 357.65 | 359.54 | 358.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 357.20 | 359.07 | 358.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:15:00 | 357.20 | 359.07 | 358.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 357.40 | 358.74 | 358.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 13:15:00 | 357.40 | 358.74 | 358.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 360.00 | 359.05 | 358.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 15:15:00 | 360.00 | 359.05 | 358.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 362.05 | 360.03 | 359.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:15:00 | 362.05 | 360.03 | 359.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 358.45 | 359.94 | 359.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:15:00 | 358.45 | 359.94 | 359.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 358.00 | 359.55 | 359.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:15:00 | 358.00 | 359.55 | 359.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 15:15:00 | 355.00 | 358.64 | 358.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 10:15:00 | 353.05 | 357.39 | 358.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 336.60 | 334.37 | 340.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 14:15:00 | 336.60 | 334.37 | 340.25 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 353.00 | 338.39 | 341.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 353.00 | 338.39 | 341.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 351.70 | 341.05 | 342.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:15:00 | 351.70 | 341.05 | 342.04 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 349.95 | 342.83 | 342.76 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 15:15:00 | 342.00 | 342.76 | 342.79 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 346.10 | 343.43 | 343.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 09:15:00 | 352.20 | 349.69 | 347.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 347.10 | 349.17 | 347.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 10:15:00 | 347.10 | 349.17 | 347.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 347.10 | 349.17 | 347.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:15:00 | 347.10 | 349.17 | 347.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 347.00 | 348.74 | 347.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 11:15:00 | 347.00 | 348.74 | 347.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 347.05 | 348.40 | 347.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:15:00 | 347.05 | 348.40 | 347.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 349.25 | 348.57 | 347.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:15:00 | 349.25 | 348.57 | 347.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 348.00 | 348.46 | 347.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 14:15:00 | 348.00 | 348.46 | 347.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 345.45 | 347.85 | 347.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:15:00 | 345.45 | 347.85 | 347.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 343.40 | 346.96 | 346.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 343.40 | 346.96 | 346.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 341.80 | 345.93 | 346.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 339.95 | 343.77 | 345.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 344.50 | 339.59 | 341.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 344.50 | 339.59 | 341.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 344.50 | 339.59 | 341.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 344.50 | 339.59 | 341.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 345.45 | 340.76 | 341.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:15:00 | 345.45 | 340.76 | 341.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 349.10 | 343.55 | 342.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 09:15:00 | 378.70 | 352.35 | 347.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 13:15:00 | 363.00 | 364.69 | 356.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-25 13:15:00 | 363.00 | 364.69 | 356.04 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 359.40 | 363.63 | 356.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:15:00 | 359.40 | 363.63 | 356.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 355.50 | 362.00 | 356.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:15:00 | 355.50 | 362.00 | 356.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 361.95 | 361.99 | 356.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 361.95 | 361.99 | 356.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 360.90 | 361.22 | 358.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 14:15:00 | 360.90 | 361.22 | 358.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 358.95 | 360.77 | 358.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:15:00 | 358.95 | 360.77 | 358.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 357.95 | 360.20 | 358.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 357.95 | 360.20 | 358.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 358.50 | 359.86 | 358.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:15:00 | 358.50 | 359.86 | 358.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 356.35 | 359.16 | 358.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:15:00 | 356.35 | 359.16 | 358.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 357.50 | 358.83 | 358.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:15:00 | 357.50 | 358.83 | 358.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 358.70 | 358.91 | 358.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 14:15:00 | 358.70 | 358.91 | 358.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 358.40 | 358.81 | 358.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 15:15:00 | 358.40 | 358.81 | 358.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 360.00 | 360.13 | 359.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:15:00 | 360.00 | 360.13 | 359.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 357.05 | 359.52 | 359.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:15:00 | 357.05 | 359.52 | 359.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 356.50 | 358.91 | 358.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:15:00 | 356.50 | 358.91 | 358.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 356.00 | 358.33 | 358.55 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 360.35 | 358.65 | 358.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 13:15:00 | 363.50 | 359.62 | 359.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 360.40 | 361.26 | 360.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 360.40 | 361.26 | 360.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 360.40 | 361.26 | 360.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 360.40 | 361.26 | 360.09 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 361.80 | 362.49 | 361.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:15:00 | 361.80 | 362.49 | 361.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 357.70 | 361.45 | 360.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:15:00 | 357.70 | 361.45 | 360.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 355.05 | 359.61 | 360.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 351.60 | 358.01 | 359.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 353.40 | 352.77 | 355.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:15:00 | 353.40 | 352.77 | 355.15 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 353.50 | 353.10 | 354.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 12:15:00 | 353.50 | 353.10 | 354.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 355.25 | 353.53 | 354.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:15:00 | 355.25 | 353.53 | 354.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 354.35 | 353.69 | 354.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:15:00 | 354.35 | 353.69 | 354.74 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 354.65 | 353.88 | 354.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:15:00 | 354.65 | 353.88 | 354.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 350.95 | 350.29 | 352.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:15:00 | 350.95 | 350.29 | 352.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 348.55 | 349.45 | 351.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 15:15:00 | 348.55 | 349.45 | 351.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 349.80 | 349.52 | 351.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:15:00 | 349.80 | 349.52 | 351.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 350.75 | 349.77 | 351.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:15:00 | 350.75 | 349.77 | 351.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 348.75 | 349.56 | 351.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:15:00 | 348.75 | 349.56 | 351.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 350.55 | 348.59 | 349.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 350.55 | 348.59 | 349.89 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 354.05 | 349.68 | 350.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:15:00 | 354.05 | 349.68 | 350.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 12:15:00 | 353.40 | 350.94 | 350.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 14:15:00 | 354.90 | 352.12 | 351.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 11:15:00 | 351.80 | 352.88 | 352.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 11:15:00 | 351.80 | 352.88 | 352.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 351.80 | 352.88 | 352.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 11:15:00 | 351.80 | 352.88 | 352.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 352.30 | 352.76 | 352.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 12:15:00 | 352.30 | 352.76 | 352.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 351.30 | 352.47 | 352.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 13:15:00 | 351.30 | 352.47 | 352.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 351.90 | 352.36 | 351.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 14:15:00 | 351.90 | 352.36 | 351.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 352.00 | 352.28 | 352.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 15:15:00 | 352.00 | 352.28 | 352.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 353.05 | 352.44 | 352.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 353.05 | 352.44 | 352.09 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 10:15:00 | 346.30 | 351.21 | 351.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 11:15:00 | 345.00 | 349.97 | 350.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 348.20 | 343.95 | 345.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 348.20 | 343.95 | 345.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 348.20 | 343.95 | 345.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:15:00 | 348.20 | 343.95 | 345.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 349.40 | 345.04 | 346.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:15:00 | 349.40 | 345.04 | 346.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 344.90 | 345.72 | 346.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:15:00 | 344.90 | 345.72 | 346.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 345.95 | 345.76 | 346.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:15:00 | 345.95 | 345.76 | 346.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 346.90 | 345.99 | 346.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 15:15:00 | 346.90 | 345.99 | 346.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 349.40 | 346.67 | 346.56 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 344.90 | 346.18 | 346.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 15:15:00 | 343.25 | 344.97 | 345.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 348.95 | 344.36 | 344.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 348.95 | 344.36 | 344.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 348.95 | 344.36 | 344.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 348.95 | 344.36 | 344.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 10:15:00 | 347.90 | 345.07 | 344.99 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 344.30 | 344.91 | 344.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 13:15:00 | 342.90 | 344.31 | 344.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 310.40 | 309.15 | 315.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 14:15:00 | 310.40 | 309.15 | 315.87 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 320.50 | 311.64 | 315.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 320.50 | 311.64 | 315.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 319.35 | 313.19 | 316.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:15:00 | 319.35 | 313.19 | 316.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 322.45 | 318.12 | 317.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 326.70 | 320.22 | 318.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 321.05 | 321.95 | 320.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 321.05 | 321.95 | 320.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 321.05 | 321.95 | 320.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 14:15:00 | 321.05 | 321.95 | 320.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 320.40 | 321.64 | 320.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:15:00 | 320.40 | 321.64 | 320.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 323.70 | 322.05 | 320.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:15:00 | 323.70 | 322.05 | 320.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 321.85 | 322.01 | 320.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:15:00 | 321.85 | 322.01 | 320.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 322.10 | 322.03 | 320.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:15:00 | 322.10 | 322.03 | 320.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 319.00 | 321.42 | 320.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:15:00 | 319.00 | 321.42 | 320.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 320.05 | 321.15 | 320.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:15:00 | 320.05 | 321.15 | 320.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 321.15 | 321.15 | 320.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:15:00 | 321.15 | 321.15 | 320.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 320.55 | 321.03 | 320.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:15:00 | 320.55 | 321.03 | 320.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 326.60 | 322.47 | 321.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:15:00 | 326.60 | 322.47 | 321.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 345.10 | 343.97 | 340.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 345.10 | 343.97 | 340.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 339.55 | 342.71 | 340.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:15:00 | 339.55 | 342.71 | 340.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 337.70 | 341.71 | 340.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:15:00 | 337.70 | 341.71 | 340.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-11-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 14:15:00 | 331.80 | 339.00 | 339.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 15:15:00 | 331.00 | 337.40 | 338.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 338.95 | 337.71 | 338.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 338.95 | 337.71 | 338.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 338.95 | 337.71 | 338.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 338.95 | 337.71 | 338.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 338.40 | 337.85 | 338.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:15:00 | 338.40 | 337.85 | 338.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 338.90 | 338.06 | 338.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 11:15:00 | 338.90 | 338.06 | 338.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 339.00 | 338.25 | 338.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:15:00 | 339.00 | 338.25 | 338.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 338.05 | 338.21 | 338.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:15:00 | 338.05 | 338.21 | 338.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 332.50 | 337.07 | 338.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:15:00 | 332.50 | 337.07 | 338.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 336.70 | 336.26 | 337.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:15:00 | 336.70 | 336.26 | 337.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 334.50 | 334.58 | 335.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 334.50 | 334.58 | 335.89 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 337.00 | 335.07 | 335.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:15:00 | 337.00 | 335.07 | 335.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 336.10 | 335.27 | 336.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 11:15:00 | 336.10 | 335.27 | 336.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 12:15:00 | 334.20 | 335.06 | 335.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 12:15:00 | 334.20 | 335.06 | 335.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 330.35 | 333.57 | 334.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:15:00 | 330.35 | 333.57 | 334.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 334.65 | 330.12 | 331.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 334.65 | 330.12 | 331.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 335.60 | 331.22 | 332.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:15:00 | 335.60 | 331.22 | 332.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 332.00 | 331.45 | 332.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:15:00 | 332.00 | 331.45 | 332.18 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 13:15:00 | 332.50 | 331.66 | 332.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 13:15:00 | 332.50 | 331.66 | 332.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 332.90 | 331.91 | 332.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 14:15:00 | 332.90 | 331.91 | 332.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 332.30 | 331.99 | 332.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:15:00 | 332.30 | 331.99 | 332.28 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 337.40 | 333.07 | 332.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 10:15:00 | 346.40 | 335.74 | 333.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 12:15:00 | 338.80 | 340.51 | 338.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-21 12:15:00 | 338.80 | 340.51 | 338.35 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 338.00 | 339.95 | 338.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 14:15:00 | 338.00 | 339.95 | 338.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 338.40 | 339.64 | 338.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 15:15:00 | 338.40 | 339.64 | 338.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 338.55 | 339.42 | 338.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 338.55 | 339.42 | 338.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 337.50 | 339.04 | 338.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:15:00 | 337.50 | 339.04 | 338.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 335.05 | 338.24 | 338.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:15:00 | 335.05 | 338.24 | 338.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 334.55 | 337.50 | 337.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 11:15:00 | 332.35 | 334.79 | 336.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 334.85 | 333.36 | 334.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 334.85 | 333.36 | 334.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 334.85 | 333.36 | 334.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:15:00 | 334.85 | 333.36 | 334.74 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 335.45 | 333.78 | 334.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:15:00 | 335.45 | 333.78 | 334.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 334.05 | 333.83 | 334.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:15:00 | 334.05 | 333.83 | 334.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 334.05 | 333.88 | 334.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:15:00 | 334.05 | 333.88 | 334.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 334.70 | 334.04 | 334.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:15:00 | 334.70 | 334.04 | 334.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 334.20 | 334.07 | 334.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:15:00 | 334.20 | 334.07 | 334.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 334.30 | 334.12 | 334.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 15:15:00 | 334.30 | 334.12 | 334.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 335.90 | 334.47 | 334.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 335.90 | 334.47 | 334.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 335.65 | 334.71 | 334.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:15:00 | 335.65 | 334.71 | 334.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 337.10 | 335.19 | 335.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 338.50 | 335.99 | 335.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 10:15:00 | 335.70 | 335.93 | 335.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 10:15:00 | 335.70 | 335.93 | 335.52 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 335.20 | 335.79 | 335.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:15:00 | 335.20 | 335.79 | 335.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 335.00 | 335.63 | 335.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 12:15:00 | 335.00 | 335.63 | 335.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 334.65 | 335.43 | 335.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 13:15:00 | 334.65 | 335.43 | 335.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 334.95 | 335.34 | 335.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:15:00 | 334.95 | 335.34 | 335.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 15:15:00 | 333.25 | 334.92 | 335.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 13:15:00 | 331.45 | 333.51 | 334.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 335.10 | 333.56 | 334.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 335.10 | 333.56 | 334.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 335.10 | 333.56 | 334.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:15:00 | 335.10 | 333.56 | 334.11 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 341.50 | 335.15 | 334.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 344.20 | 339.61 | 337.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 14:15:00 | 341.60 | 341.99 | 339.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 14:15:00 | 341.60 | 341.99 | 339.71 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 339.80 | 342.12 | 340.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:15:00 | 339.80 | 342.12 | 340.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 340.75 | 341.84 | 340.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:15:00 | 340.75 | 341.84 | 340.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 339.00 | 341.27 | 340.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 13:15:00 | 339.00 | 341.27 | 340.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 339.50 | 340.92 | 340.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 14:15:00 | 339.50 | 340.92 | 340.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 344.90 | 341.72 | 340.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 15:15:00 | 344.90 | 341.72 | 340.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 341.15 | 342.26 | 341.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:15:00 | 341.15 | 342.26 | 341.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 338.90 | 341.59 | 341.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:15:00 | 338.90 | 341.59 | 341.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 340.10 | 341.29 | 341.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:15:00 | 340.10 | 341.29 | 341.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 15:15:00 | 339.55 | 340.64 | 340.74 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 343.00 | 341.11 | 340.95 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 11:15:00 | 339.25 | 340.63 | 340.75 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 09:15:00 | 362.60 | 344.82 | 342.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 10:15:00 | 375.85 | 351.02 | 345.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 13:15:00 | 386.00 | 387.19 | 378.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 13:15:00 | 386.00 | 387.19 | 378.27 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 420.00 | 423.70 | 415.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:15:00 | 420.00 | 423.70 | 415.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 419.80 | 424.73 | 419.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:15:00 | 419.80 | 424.73 | 419.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 421.00 | 423.99 | 419.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:15:00 | 421.00 | 423.99 | 419.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 417.75 | 422.74 | 419.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:15:00 | 417.75 | 422.74 | 419.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 417.05 | 421.60 | 419.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:15:00 | 417.05 | 421.60 | 419.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 418.45 | 420.97 | 419.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 14:15:00 | 418.45 | 420.97 | 419.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 420.00 | 420.78 | 419.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 15:15:00 | 420.00 | 420.78 | 419.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 419.00 | 420.42 | 419.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:15:00 | 419.00 | 420.42 | 419.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 416.70 | 419.68 | 419.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:15:00 | 416.70 | 419.68 | 419.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 416.25 | 418.99 | 418.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:15:00 | 416.25 | 418.99 | 418.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 13:15:00 | 416.10 | 418.25 | 418.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 14:15:00 | 406.00 | 415.80 | 417.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 416.00 | 414.73 | 416.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 416.00 | 414.73 | 416.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 416.00 | 414.73 | 416.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:15:00 | 416.00 | 414.73 | 416.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 409.85 | 413.76 | 415.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:15:00 | 409.85 | 413.76 | 415.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 407.90 | 402.08 | 406.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:15:00 | 407.90 | 402.08 | 406.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 404.90 | 402.64 | 406.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 14:15:00 | 404.90 | 402.64 | 406.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 407.00 | 403.51 | 406.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:15:00 | 407.00 | 403.51 | 406.11 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 409.65 | 404.74 | 406.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 409.65 | 404.74 | 406.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 405.55 | 404.90 | 406.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:15:00 | 405.55 | 404.90 | 406.35 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 406.55 | 403.64 | 404.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 406.55 | 403.64 | 404.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 405.10 | 403.93 | 404.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:15:00 | 405.10 | 403.93 | 404.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 11:15:00 | 405.40 | 404.23 | 404.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 11:15:00 | 405.40 | 404.23 | 404.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 403.90 | 404.16 | 404.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 12:15:00 | 403.90 | 404.16 | 404.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 407.45 | 404.57 | 404.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 14:15:00 | 407.45 | 404.57 | 404.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 403.60 | 404.38 | 404.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 15:15:00 | 403.60 | 404.38 | 404.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 418.05 | 407.11 | 405.97 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 13:15:00 | 405.80 | 408.50 | 408.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 15:15:00 | 404.10 | 407.11 | 407.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 409.95 | 406.70 | 407.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 10:15:00 | 409.95 | 406.70 | 407.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 409.95 | 406.70 | 407.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:15:00 | 409.95 | 406.70 | 407.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 409.85 | 407.33 | 407.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:15:00 | 409.85 | 407.33 | 407.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 405.50 | 406.57 | 407.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 13:15:00 | 405.50 | 406.57 | 407.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 407.50 | 406.76 | 407.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 14:15:00 | 407.50 | 406.76 | 407.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 15:15:00 | 409.20 | 407.25 | 407.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 15:15:00 | 409.20 | 407.25 | 407.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 405.50 | 406.90 | 407.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:15:00 | 405.50 | 406.90 | 407.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 406.20 | 406.43 | 406.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:15:00 | 406.20 | 406.43 | 406.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 407.05 | 406.55 | 407.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:15:00 | 407.05 | 406.55 | 407.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 403.35 | 405.91 | 406.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 14:15:00 | 403.35 | 405.91 | 406.66 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 401.40 | 404.70 | 405.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:15:00 | 401.40 | 404.70 | 405.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 399.80 | 400.25 | 402.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:15:00 | 399.80 | 400.25 | 402.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 403.00 | 401.00 | 402.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:15:00 | 403.00 | 401.00 | 402.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 401.25 | 401.05 | 402.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 14:15:00 | 401.25 | 401.05 | 402.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 407.65 | 402.33 | 402.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:15:00 | 407.65 | 402.33 | 402.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 10:15:00 | 406.00 | 403.06 | 402.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 15:15:00 | 409.00 | 406.30 | 404.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 413.00 | 414.57 | 411.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 14:15:00 | 413.00 | 414.57 | 411.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 413.00 | 414.57 | 411.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:15:00 | 413.00 | 414.57 | 411.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 410.50 | 413.76 | 411.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:15:00 | 410.50 | 413.76 | 411.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 411.20 | 415.21 | 413.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:15:00 | 411.20 | 415.21 | 413.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 409.90 | 414.15 | 413.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:15:00 | 409.90 | 414.15 | 413.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 410.70 | 413.46 | 412.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 15:15:00 | 410.70 | 413.46 | 412.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 416.75 | 413.97 | 413.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:15:00 | 416.75 | 413.97 | 413.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 433.55 | 435.88 | 432.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 433.55 | 435.88 | 432.85 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 430.85 | 434.73 | 432.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:15:00 | 430.85 | 434.73 | 432.85 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 428.80 | 433.55 | 432.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 12:15:00 | 428.80 | 433.55 | 432.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 433.00 | 433.13 | 432.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:15:00 | 433.00 | 433.13 | 432.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 429.50 | 432.40 | 432.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:15:00 | 429.50 | 432.40 | 432.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 10:15:00 | 430.65 | 432.05 | 432.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 427.55 | 430.87 | 431.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 428.25 | 426.41 | 428.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 428.25 | 426.41 | 428.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 428.25 | 426.41 | 428.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:15:00 | 428.25 | 426.41 | 428.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 430.70 | 427.27 | 428.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:15:00 | 430.70 | 427.27 | 428.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 433.50 | 428.51 | 428.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:15:00 | 433.50 | 428.51 | 428.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 432.80 | 429.37 | 429.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 436.00 | 431.23 | 430.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 440.95 | 440.99 | 436.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 11:15:00 | 440.95 | 440.99 | 436.33 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 431.60 | 439.20 | 436.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:15:00 | 431.60 | 439.20 | 436.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 426.35 | 436.63 | 435.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:15:00 | 426.35 | 436.63 | 435.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 424.60 | 434.23 | 434.44 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 454.00 | 437.78 | 435.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 463.00 | 449.66 | 442.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 11:15:00 | 466.90 | 469.24 | 461.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 11:15:00 | 466.90 | 469.24 | 461.38 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 475.80 | 471.79 | 465.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 475.80 | 471.79 | 465.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 472.60 | 471.86 | 468.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 14:15:00 | 472.60 | 471.86 | 468.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 466.45 | 470.56 | 468.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 11:15:00 | 466.45 | 470.56 | 468.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 467.75 | 470.00 | 468.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:15:00 | 467.75 | 470.00 | 468.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 469.05 | 469.81 | 468.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:15:00 | 469.05 | 469.81 | 468.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 468.65 | 469.58 | 468.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:15:00 | 468.65 | 469.58 | 468.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 477.00 | 471.06 | 469.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:15:00 | 477.00 | 471.06 | 469.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 469.50 | 474.26 | 472.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:15:00 | 469.50 | 474.26 | 472.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 467.80 | 472.97 | 471.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:15:00 | 467.80 | 472.97 | 471.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 459.00 | 470.17 | 470.56 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 14:15:00 | 478.10 | 470.37 | 469.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 09:15:00 | 488.20 | 474.40 | 471.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 561.30 | 584.00 | 564.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 561.30 | 584.00 | 564.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 561.30 | 584.00 | 564.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:15:00 | 561.30 | 584.00 | 564.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 566.00 | 580.40 | 565.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:15:00 | 566.00 | 580.40 | 565.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 560.55 | 576.43 | 564.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 11:15:00 | 560.55 | 576.43 | 564.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 563.90 | 573.93 | 564.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 12:15:00 | 563.90 | 573.93 | 564.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 564.25 | 571.99 | 564.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:15:00 | 564.25 | 571.99 | 564.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 560.80 | 569.75 | 564.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:15:00 | 560.80 | 569.75 | 564.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 560.00 | 567.80 | 563.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:15:00 | 560.00 | 567.80 | 563.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 559.45 | 566.13 | 563.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 559.45 | 566.13 | 563.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 554.55 | 563.82 | 562.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:15:00 | 554.55 | 563.82 | 562.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 548.15 | 560.68 | 561.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 12:15:00 | 545.55 | 557.66 | 559.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 10:15:00 | 560.45 | 554.97 | 557.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 560.45 | 554.97 | 557.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 560.45 | 554.97 | 557.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:15:00 | 560.45 | 554.97 | 557.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 551.40 | 554.26 | 556.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:15:00 | 551.40 | 554.26 | 556.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 554.95 | 554.02 | 556.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:15:00 | 554.95 | 554.02 | 556.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 551.75 | 553.57 | 555.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:15:00 | 551.75 | 553.57 | 555.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 566.90 | 556.23 | 556.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:15:00 | 566.90 | 556.23 | 556.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 581.15 | 561.22 | 559.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 10:15:00 | 583.70 | 565.71 | 561.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 559.95 | 571.34 | 567.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 559.95 | 571.34 | 567.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 559.95 | 571.34 | 567.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:15:00 | 559.95 | 571.34 | 567.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 564.90 | 570.05 | 566.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:15:00 | 564.90 | 570.05 | 566.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 559.80 | 566.87 | 565.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:15:00 | 559.80 | 566.87 | 565.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 14:15:00 | 558.00 | 563.99 | 564.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 09:15:00 | 551.00 | 560.59 | 563.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 562.80 | 560.13 | 562.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 11:15:00 | 562.80 | 560.13 | 562.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 562.80 | 560.13 | 562.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:15:00 | 562.80 | 560.13 | 562.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 564.80 | 561.06 | 562.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:15:00 | 564.80 | 561.06 | 562.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 564.10 | 561.67 | 562.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:15:00 | 564.10 | 561.67 | 562.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 561.00 | 561.54 | 562.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:15:00 | 561.00 | 561.54 | 562.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 560.00 | 561.23 | 562.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:15:00 | 560.00 | 561.23 | 562.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 560.00 | 560.98 | 562.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:15:00 | 560.00 | 560.98 | 562.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 564.80 | 554.46 | 556.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:15:00 | 564.80 | 554.46 | 556.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 564.75 | 556.51 | 557.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 12:15:00 | 564.75 | 556.51 | 557.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 558.00 | 556.77 | 557.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:15:00 | 558.00 | 556.77 | 557.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 553.05 | 556.03 | 556.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 15:15:00 | 553.05 | 556.03 | 556.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 546.05 | 554.03 | 555.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 546.05 | 554.03 | 555.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 557.45 | 554.71 | 555.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:15:00 | 557.45 | 554.71 | 555.98 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 554.10 | 554.59 | 555.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:15:00 | 554.10 | 554.59 | 555.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 560.35 | 555.74 | 556.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:15:00 | 560.35 | 555.74 | 556.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 558.50 | 556.29 | 556.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:15:00 | 558.50 | 556.29 | 556.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 553.40 | 555.72 | 556.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:15:00 | 553.40 | 555.72 | 556.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 565.10 | 557.59 | 556.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 13:15:00 | 575.20 | 564.79 | 561.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 575.00 | 577.23 | 571.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 14:15:00 | 575.00 | 577.23 | 571.05 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 583.00 | 586.10 | 580.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 583.00 | 586.10 | 580.56 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 582.00 | 585.28 | 580.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:15:00 | 582.00 | 585.28 | 580.69 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 585.95 | 585.41 | 581.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:15:00 | 585.95 | 585.41 | 581.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 579.95 | 584.32 | 581.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:15:00 | 579.95 | 584.32 | 581.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 599.00 | 587.26 | 582.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:15:00 | 599.00 | 587.26 | 582.69 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 601.00 | 599.07 | 590.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:15:00 | 601.00 | 599.07 | 590.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 620.00 | 625.45 | 614.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 14:15:00 | 620.00 | 625.45 | 614.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 622.75 | 624.91 | 615.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:15:00 | 622.75 | 624.91 | 615.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 694.00 | 704.28 | 694.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:15:00 | 694.00 | 704.28 | 694.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 703.95 | 704.22 | 695.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:15:00 | 703.95 | 704.22 | 695.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 682.00 | 699.77 | 693.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:15:00 | 682.00 | 699.77 | 693.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 690.40 | 697.90 | 693.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:15:00 | 690.40 | 697.90 | 693.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 685.00 | 695.32 | 692.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:15:00 | 685.00 | 695.32 | 692.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 678.20 | 689.12 | 690.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 674.60 | 686.22 | 688.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 630.00 | 606.26 | 623.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 630.00 | 606.26 | 623.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 630.00 | 606.26 | 623.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:15:00 | 630.00 | 606.26 | 623.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 624.10 | 609.83 | 623.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:15:00 | 624.10 | 609.83 | 623.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 605.50 | 608.96 | 622.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 605.50 | 608.96 | 622.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 616.00 | 609.53 | 618.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:15:00 | 616.00 | 609.53 | 618.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 625.20 | 612.67 | 618.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:15:00 | 625.20 | 612.67 | 618.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 618.80 | 613.89 | 618.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:15:00 | 618.80 | 613.89 | 618.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 615.00 | 614.11 | 618.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:15:00 | 615.00 | 614.11 | 618.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 605.05 | 612.30 | 617.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:15:00 | 605.05 | 612.30 | 617.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 609.95 | 611.78 | 616.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:15:00 | 609.95 | 611.78 | 616.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 600.00 | 605.91 | 611.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:15:00 | 600.00 | 605.91 | 611.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 585.00 | 590.30 | 599.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:15:00 | 585.00 | 590.30 | 599.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 581.30 | 586.89 | 594.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:15:00 | 581.30 | 586.89 | 594.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 594.00 | 588.31 | 594.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 15:15:00 | 594.00 | 588.31 | 594.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 609.25 | 592.50 | 595.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:15:00 | 609.25 | 592.50 | 595.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 619.15 | 597.83 | 597.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 631.30 | 616.06 | 608.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 12:15:00 | 800.05 | 823.25 | 797.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 12:15:00 | 800.05 | 823.25 | 797.28 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 795.00 | 817.60 | 797.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:15:00 | 795.00 | 817.60 | 797.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 794.90 | 813.06 | 796.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:15:00 | 794.90 | 813.06 | 796.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 799.20 | 810.29 | 797.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:15:00 | 799.20 | 810.29 | 797.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 809.00 | 810.03 | 798.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:15:00 | 809.00 | 810.03 | 798.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 800.00 | 808.02 | 798.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:15:00 | 800.00 | 808.02 | 798.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 808.00 | 808.02 | 799.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:15:00 | 808.00 | 808.02 | 799.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 802.90 | 805.72 | 799.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:15:00 | 802.90 | 805.72 | 799.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 808.00 | 806.18 | 800.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:15:00 | 808.00 | 806.18 | 800.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 794.00 | 803.87 | 800.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:15:00 | 794.00 | 803.87 | 800.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 800.40 | 803.18 | 800.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:15:00 | 800.40 | 803.18 | 800.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 802.00 | 802.94 | 800.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:15:00 | 802.00 | 802.94 | 800.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 790.00 | 800.35 | 799.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:15:00 | 790.00 | 800.35 | 799.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 789.60 | 798.20 | 798.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 14:15:00 | 787.00 | 795.96 | 797.58 | Break + close below crossover candle low |

### Cycle 64 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 823.00 | 800.10 | 799.10 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 798.00 | 806.85 | 807.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 11:15:00 | 772.00 | 795.21 | 801.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 09:15:00 | 778.90 | 771.71 | 780.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 09:15:00 | 778.90 | 771.71 | 780.43 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 786.00 | 769.24 | 774.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 786.00 | 769.24 | 774.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 795.05 | 774.40 | 776.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:15:00 | 795.05 | 774.40 | 776.28 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 11:15:00 | 795.05 | 778.53 | 777.99 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 10:15:00 | 760.00 | 776.97 | 778.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 725.00 | 762.08 | 770.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 10:15:00 | 738.00 | 729.26 | 745.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-22 10:15:00 | 738.00 | 729.26 | 745.29 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 748.50 | 733.03 | 744.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:15:00 | 748.50 | 733.03 | 744.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 13:15:00 | 729.85 | 732.40 | 742.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 13:15:00 | 729.85 | 732.40 | 742.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 742.00 | 734.32 | 742.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 14:15:00 | 742.00 | 734.32 | 742.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 744.00 | 736.25 | 742.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:15:00 | 744.00 | 736.25 | 742.93 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 747.95 | 738.59 | 743.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:15:00 | 747.95 | 738.59 | 743.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 758.95 | 742.66 | 744.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:15:00 | 758.95 | 742.66 | 744.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 754.00 | 744.93 | 745.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:15:00 | 754.00 | 744.93 | 745.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 12:15:00 | 752.90 | 746.52 | 746.30 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 727.00 | 742.78 | 744.65 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 750.00 | 741.18 | 740.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 760.00 | 747.33 | 744.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 11:15:00 | 796.00 | 803.85 | 792.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 11:15:00 | 796.00 | 803.85 | 792.20 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 808.00 | 812.98 | 803.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:15:00 | 808.00 | 812.98 | 803.84 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 805.40 | 811.47 | 803.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:15:00 | 805.40 | 811.47 | 803.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 820.05 | 813.18 | 805.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:15:00 | 820.05 | 813.18 | 805.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 821.80 | 814.91 | 806.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 14:15:00 | 821.80 | 814.91 | 806.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 785.55 | 810.03 | 806.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 785.55 | 810.03 | 806.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 809.90 | 810.00 | 806.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:15:00 | 809.90 | 810.00 | 806.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 807.35 | 809.47 | 806.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:15:00 | 807.35 | 809.47 | 806.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 818.00 | 811.18 | 807.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:15:00 | 818.00 | 811.18 | 807.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 823.10 | 820.92 | 814.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:15:00 | 823.10 | 820.92 | 814.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 800.00 | 816.74 | 813.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:15:00 | 800.00 | 816.74 | 813.30 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 793.40 | 812.07 | 811.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:15:00 | 793.40 | 812.07 | 811.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 13:15:00 | 794.00 | 808.46 | 809.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 10:15:00 | 792.00 | 800.81 | 805.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 810.00 | 802.65 | 805.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 11:15:00 | 810.00 | 802.65 | 805.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 810.00 | 802.65 | 805.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:15:00 | 810.00 | 802.65 | 805.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 819.70 | 806.06 | 807.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:15:00 | 819.70 | 806.06 | 807.12 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 820.00 | 808.85 | 808.29 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 794.95 | 806.97 | 807.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 782.40 | 799.89 | 803.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 12:15:00 | 798.00 | 797.72 | 801.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 12:15:00 | 798.00 | 797.72 | 801.71 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 802.90 | 798.76 | 801.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:15:00 | 802.90 | 798.76 | 801.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 788.00 | 796.61 | 800.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:15:00 | 788.00 | 796.61 | 800.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 784.80 | 784.35 | 790.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:15:00 | 784.80 | 784.35 | 790.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 810.00 | 789.48 | 791.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 810.00 | 789.48 | 791.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 814.00 | 794.38 | 793.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 822.00 | 799.91 | 796.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 805.00 | 815.72 | 809.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 13:15:00 | 805.00 | 815.72 | 809.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 805.00 | 815.72 | 809.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:15:00 | 805.00 | 815.72 | 809.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 814.00 | 815.38 | 809.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:15:00 | 814.00 | 815.38 | 809.82 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 812.00 | 814.00 | 810.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 812.00 | 814.00 | 810.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 811.95 | 813.59 | 810.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:15:00 | 811.95 | 813.59 | 810.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 815.40 | 813.95 | 810.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:15:00 | 815.40 | 813.95 | 810.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 825.80 | 816.32 | 812.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:15:00 | 825.80 | 816.32 | 812.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 903.20 | 895.01 | 875.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 903.20 | 895.01 | 875.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 870.90 | 909.96 | 904.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 870.90 | 909.96 | 904.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 870.90 | 895.90 | 898.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 827.35 | 870.38 | 884.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 697.80 | 697.35 | 727.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:15:00 | 697.80 | 697.35 | 727.25 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 719.70 | 698.05 | 717.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 719.70 | 698.05 | 717.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 719.70 | 702.38 | 717.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:15:00 | 719.70 | 702.38 | 717.96 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 719.70 | 708.61 | 718.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:15:00 | 719.70 | 708.61 | 718.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 696.35 | 674.63 | 686.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:15:00 | 696.35 | 674.63 | 686.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 694.70 | 678.64 | 687.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:15:00 | 694.70 | 678.64 | 687.12 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 716.75 | 686.26 | 689.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 716.75 | 686.26 | 689.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 699.70 | 692.91 | 692.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 717.40 | 698.52 | 695.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 815.25 | 818.39 | 797.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:15:00 | 815.25 | 818.39 | 797.28 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 802.05 | 810.68 | 800.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 802.05 | 810.68 | 800.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 793.90 | 807.32 | 799.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:15:00 | 793.90 | 807.32 | 799.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 790.85 | 804.03 | 798.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:15:00 | 790.85 | 804.03 | 798.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 791.00 | 795.75 | 795.98 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 814.00 | 799.40 | 797.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 848.80 | 814.46 | 806.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 904.00 | 918.57 | 898.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:15:00 | 904.00 | 918.57 | 898.74 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 905.00 | 915.85 | 899.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:15:00 | 905.00 | 915.85 | 899.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 898.50 | 912.21 | 900.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:15:00 | 898.50 | 912.21 | 900.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 906.40 | 911.05 | 901.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:15:00 | 906.40 | 911.05 | 901.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 900.50 | 907.62 | 901.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:15:00 | 900.50 | 907.62 | 901.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 900.70 | 906.24 | 901.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 900.70 | 906.24 | 901.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 901.85 | 905.36 | 901.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:15:00 | 901.85 | 905.36 | 901.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 900.40 | 904.37 | 901.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:15:00 | 900.40 | 904.37 | 901.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 897.55 | 903.01 | 900.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:15:00 | 897.55 | 903.01 | 900.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 887.15 | 900.03 | 899.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:15:00 | 887.15 | 900.03 | 899.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 889.00 | 897.83 | 898.84 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 899.90 | 897.48 | 897.42 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 896.70 | 897.32 | 897.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 880.15 | 893.89 | 895.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 909.40 | 886.99 | 889.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 909.40 | 886.99 | 889.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 909.40 | 886.99 | 889.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 909.40 | 886.99 | 889.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 919.40 | 893.47 | 892.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 945.95 | 917.24 | 909.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 922.75 | 931.56 | 924.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 11:15:00 | 922.75 | 931.56 | 924.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 922.75 | 931.56 | 924.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:15:00 | 922.75 | 931.56 | 924.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 918.00 | 928.85 | 923.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:15:00 | 918.00 | 928.85 | 923.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 914.00 | 925.88 | 923.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:15:00 | 914.00 | 925.88 | 923.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 920.00 | 924.14 | 922.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:15:00 | 920.00 | 924.14 | 922.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 915.00 | 922.31 | 922.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 915.00 | 922.31 | 922.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 916.90 | 921.23 | 921.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 893.70 | 914.13 | 918.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 15:15:00 | 886.80 | 885.61 | 894.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:15:00 | 886.80 | 885.61 | 894.71 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 852.95 | 879.08 | 890.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 852.95 | 879.08 | 890.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 875.50 | 867.22 | 877.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 875.50 | 867.22 | 877.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 872.00 | 869.21 | 876.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:15:00 | 872.00 | 869.21 | 876.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 880.65 | 871.50 | 877.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:15:00 | 880.65 | 871.50 | 877.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 868.65 | 870.93 | 876.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:15:00 | 868.65 | 870.93 | 876.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 875.00 | 868.15 | 873.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 875.00 | 868.15 | 873.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 850.00 | 864.52 | 871.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:15:00 | 850.00 | 864.52 | 871.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 781.05 | 765.37 | 782.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 781.05 | 765.37 | 782.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 763.85 | 765.07 | 781.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:15:00 | 763.85 | 765.07 | 781.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 786.75 | 768.90 | 777.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:15:00 | 786.75 | 768.90 | 777.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 789.00 | 772.92 | 778.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:15:00 | 789.00 | 772.92 | 778.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 770.05 | 768.24 | 772.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 770.05 | 768.24 | 772.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 764.95 | 767.58 | 771.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:15:00 | 764.95 | 767.58 | 771.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 763.00 | 761.71 | 766.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:15:00 | 763.00 | 761.71 | 766.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 764.75 | 762.43 | 765.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:15:00 | 764.75 | 762.43 | 765.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 771.25 | 764.19 | 765.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:15:00 | 771.25 | 764.19 | 765.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 766.60 | 764.67 | 765.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:15:00 | 766.60 | 764.67 | 765.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 795.00 | 770.74 | 768.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 796.00 | 779.11 | 772.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 797.55 | 801.47 | 791.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:15:00 | 797.55 | 801.47 | 791.61 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 793.70 | 799.92 | 791.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:15:00 | 793.70 | 799.92 | 791.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 794.50 | 798.83 | 792.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 794.50 | 798.83 | 792.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 793.75 | 797.82 | 792.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:15:00 | 793.75 | 797.82 | 792.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 791.75 | 796.60 | 792.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:15:00 | 791.75 | 796.60 | 792.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 791.35 | 795.55 | 792.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:15:00 | 791.35 | 795.55 | 792.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 790.30 | 794.50 | 791.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:15:00 | 790.30 | 794.50 | 791.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 791.75 | 793.95 | 791.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:15:00 | 791.75 | 793.95 | 791.91 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 799.40 | 795.04 | 792.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:15:00 | 799.40 | 795.04 | 792.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 779.00 | 797.25 | 796.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 779.00 | 797.25 | 796.45 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 779.10 | 793.62 | 794.87 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 13:15:00 | 808.35 | 796.36 | 794.83 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 767.30 | 792.90 | 793.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 753.05 | 779.19 | 787.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 807.30 | 770.54 | 777.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 807.30 | 770.54 | 777.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 807.30 | 770.54 | 777.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 807.30 | 770.54 | 777.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 806.00 | 783.23 | 782.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 809.50 | 788.48 | 785.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 787.50 | 790.13 | 786.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 787.50 | 790.13 | 786.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 787.50 | 790.13 | 786.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 14:15:00 | 787.50 | 790.13 | 786.65 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 788.65 | 789.83 | 786.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:15:00 | 788.65 | 789.83 | 786.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 794.00 | 790.67 | 787.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:15:00 | 794.00 | 790.67 | 787.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 821.65 | 831.13 | 826.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:15:00 | 821.65 | 831.13 | 826.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 822.00 | 829.30 | 826.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:15:00 | 822.00 | 829.30 | 826.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 816.95 | 825.90 | 825.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:15:00 | 816.95 | 825.90 | 825.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 814.90 | 823.70 | 824.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 805.40 | 820.04 | 822.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 816.20 | 800.25 | 805.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 816.20 | 800.25 | 805.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 816.20 | 800.25 | 805.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 816.20 | 800.25 | 805.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 830.00 | 806.20 | 808.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:15:00 | 830.00 | 806.20 | 808.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 827.20 | 810.40 | 809.76 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 810.00 | 813.70 | 814.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 799.05 | 810.77 | 812.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 15:15:00 | 801.90 | 799.56 | 804.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 15:15:00 | 801.90 | 799.56 | 804.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 801.90 | 799.56 | 804.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:15:00 | 801.90 | 799.56 | 804.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 812.50 | 802.15 | 805.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 812.50 | 802.15 | 805.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 815.80 | 804.88 | 806.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:15:00 | 815.80 | 804.88 | 806.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 809.05 | 806.17 | 806.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:15:00 | 809.05 | 806.17 | 806.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 815.50 | 808.64 | 807.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 816.20 | 810.69 | 809.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 810.40 | 811.09 | 809.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:15:00 | 810.40 | 811.09 | 809.55 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 814.30 | 811.74 | 810.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:15:00 | 814.30 | 811.74 | 810.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 811.90 | 812.78 | 810.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 811.90 | 812.78 | 810.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 807.05 | 811.63 | 810.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:15:00 | 807.05 | 811.63 | 810.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 813.10 | 811.92 | 810.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 813.10 | 811.92 | 810.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 813.65 | 812.27 | 811.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:15:00 | 813.65 | 812.27 | 811.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 811.90 | 812.20 | 811.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:15:00 | 811.90 | 812.20 | 811.14 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 795.70 | 808.90 | 809.74 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 813.90 | 809.89 | 809.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 820.80 | 813.05 | 811.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 14:15:00 | 815.60 | 816.88 | 814.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 815.60 | 816.88 | 814.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 815.60 | 816.88 | 814.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:15:00 | 815.60 | 816.88 | 814.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 820.10 | 817.53 | 814.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:15:00 | 820.10 | 817.53 | 814.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 811.80 | 816.38 | 814.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 811.80 | 816.38 | 814.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 811.70 | 815.45 | 814.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:15:00 | 811.70 | 815.45 | 814.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 806.95 | 813.75 | 813.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:15:00 | 806.95 | 813.75 | 813.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 807.70 | 812.54 | 812.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 798.70 | 809.77 | 811.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 797.75 | 794.64 | 800.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 797.75 | 794.64 | 800.10 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 812.65 | 798.24 | 801.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:15:00 | 812.65 | 798.24 | 801.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 808.40 | 800.27 | 801.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:15:00 | 808.40 | 800.27 | 801.89 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 809.70 | 803.89 | 803.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 833.95 | 809.90 | 806.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 815.05 | 816.76 | 811.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:15:00 | 815.05 | 816.76 | 811.24 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 815.35 | 816.48 | 811.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:15:00 | 815.35 | 816.48 | 811.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 814.75 | 816.45 | 812.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:15:00 | 814.75 | 816.45 | 812.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 813.25 | 824.57 | 821.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:15:00 | 813.25 | 824.57 | 821.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 814.50 | 822.56 | 821.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:15:00 | 814.50 | 822.56 | 821.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 819.75 | 821.27 | 820.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:15:00 | 819.75 | 821.27 | 820.84 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 823.30 | 822.21 | 821.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:15:00 | 823.30 | 822.21 | 821.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 818.65 | 821.49 | 821.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:15:00 | 818.65 | 821.49 | 821.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 821.00 | 821.40 | 821.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:15:00 | 821.00 | 821.40 | 821.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 820.95 | 821.31 | 821.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:15:00 | 820.95 | 821.31 | 821.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 824.40 | 821.93 | 821.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:15:00 | 824.40 | 821.93 | 821.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 825.35 | 822.61 | 821.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:15:00 | 825.35 | 822.61 | 821.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 818.70 | 823.01 | 822.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:15:00 | 818.70 | 823.01 | 822.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 817.40 | 821.89 | 821.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:15:00 | 817.40 | 821.89 | 821.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 810.95 | 819.70 | 820.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 807.65 | 817.29 | 819.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 795.00 | 791.49 | 800.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 795.00 | 791.49 | 800.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 795.00 | 791.49 | 800.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 795.00 | 791.49 | 800.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 802.10 | 795.51 | 800.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:15:00 | 802.10 | 795.51 | 800.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 802.40 | 796.89 | 800.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:15:00 | 802.40 | 796.89 | 800.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 813.00 | 802.85 | 802.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 820.00 | 813.32 | 810.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 819.55 | 820.47 | 816.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:15:00 | 819.55 | 820.47 | 816.12 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 804.80 | 817.33 | 815.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 804.80 | 817.33 | 815.09 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 796.00 | 813.07 | 813.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 781.00 | 806.65 | 810.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 757.95 | 756.61 | 766.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 757.95 | 756.61 | 766.72 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 766.15 | 758.52 | 766.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:15:00 | 766.15 | 758.52 | 766.66 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 770.25 | 760.87 | 766.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:15:00 | 770.25 | 760.87 | 766.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 775.45 | 763.78 | 767.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:15:00 | 775.45 | 763.78 | 767.76 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 789.85 | 768.91 | 769.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:15:00 | 789.85 | 768.91 | 769.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 782.50 | 771.63 | 770.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 800.00 | 781.75 | 777.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 13:15:00 | 840.90 | 842.31 | 822.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:15:00 | 840.90 | 842.31 | 822.58 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 841.50 | 843.96 | 828.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 841.50 | 843.96 | 828.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 832.50 | 841.10 | 829.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:15:00 | 832.50 | 841.10 | 829.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 827.10 | 838.30 | 829.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:15:00 | 827.10 | 838.30 | 829.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 830.00 | 836.64 | 829.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:15:00 | 830.00 | 836.64 | 829.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 834.70 | 836.25 | 830.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:15:00 | 834.70 | 836.25 | 830.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 827.80 | 834.56 | 829.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:15:00 | 827.80 | 834.56 | 829.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 832.00 | 834.05 | 830.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 832.00 | 834.05 | 830.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 824.95 | 832.23 | 829.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:15:00 | 824.95 | 832.23 | 829.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 824.75 | 830.73 | 829.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:15:00 | 824.75 | 830.73 | 829.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 819.45 | 826.45 | 827.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 818.00 | 824.56 | 826.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 825.10 | 821.05 | 823.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 825.10 | 821.05 | 823.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 825.10 | 821.05 | 823.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 825.10 | 821.05 | 823.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 822.65 | 821.37 | 823.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:15:00 | 822.65 | 821.37 | 823.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 821.25 | 820.89 | 822.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:15:00 | 821.25 | 820.89 | 822.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 804.10 | 797.30 | 804.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:15:00 | 804.10 | 797.30 | 804.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 795.80 | 797.00 | 803.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:15:00 | 795.80 | 797.00 | 803.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 779.65 | 767.86 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:15:00 | 779.65 | 767.86 | 776.12 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 778.50 | 769.99 | 776.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:15:00 | 778.50 | 769.99 | 776.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 787.00 | 775.95 | 778.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:15:00 | 787.00 | 775.95 | 778.11 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 801.55 | 781.07 | 780.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 816.05 | 791.94 | 785.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 848.00 | 851.29 | 832.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:15:00 | 848.00 | 851.29 | 832.32 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 856.90 | 864.30 | 860.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:15:00 | 856.90 | 864.30 | 860.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 861.00 | 863.64 | 860.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:15:00 | 861.00 | 863.64 | 860.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 856.45 | 862.20 | 860.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:15:00 | 856.45 | 862.20 | 860.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 840.90 | 857.94 | 858.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 826.45 | 842.10 | 849.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 740.45 | 733.80 | 746.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:15:00 | 740.45 | 733.80 | 746.21 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 739.35 | 734.91 | 745.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:15:00 | 739.35 | 734.91 | 745.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 741.75 | 737.10 | 744.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:15:00 | 741.75 | 737.10 | 744.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 738.95 | 737.93 | 743.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 738.95 | 737.93 | 743.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 743.05 | 738.33 | 741.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:15:00 | 743.05 | 738.33 | 741.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 743.00 | 739.26 | 741.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:15:00 | 743.00 | 739.26 | 741.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 758.95 | 743.20 | 742.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 766.50 | 747.86 | 745.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 758.70 | 768.55 | 762.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 758.70 | 768.55 | 762.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 758.70 | 768.55 | 762.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:15:00 | 758.70 | 768.55 | 762.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 752.10 | 765.26 | 761.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:15:00 | 752.10 | 765.26 | 761.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 758.70 | 762.74 | 760.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:15:00 | 758.70 | 762.74 | 760.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 757.55 | 760.59 | 760.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:15:00 | 757.55 | 760.59 | 760.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 770.15 | 762.50 | 761.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 770.15 | 762.50 | 761.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 791.80 | 797.20 | 792.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:15:00 | 791.80 | 797.20 | 792.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 790.10 | 795.78 | 791.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 790.10 | 795.78 | 791.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 781.85 | 792.99 | 791.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:15:00 | 781.85 | 792.99 | 791.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 787.80 | 791.95 | 790.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:15:00 | 787.80 | 791.95 | 790.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 778.65 | 787.88 | 789.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 777.15 | 785.74 | 787.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 777.00 | 774.47 | 779.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 777.00 | 774.47 | 779.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 777.00 | 774.47 | 779.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 777.00 | 774.47 | 779.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 746.55 | 737.10 | 750.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 746.55 | 737.10 | 750.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 692.45 | 718.51 | 733.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 692.45 | 718.51 | 733.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 746.90 | 717.03 | 724.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 746.90 | 717.03 | 724.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 754.75 | 724.57 | 726.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:15:00 | 754.75 | 724.57 | 726.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 748.50 | 729.36 | 728.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 763.70 | 749.66 | 741.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 796.70 | 805.44 | 790.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:15:00 | 796.70 | 805.44 | 790.65 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 816.85 | 822.50 | 815.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:15:00 | 816.85 | 822.50 | 815.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 819.00 | 821.80 | 815.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:15:00 | 819.00 | 821.80 | 815.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 809.00 | 819.24 | 814.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 809.00 | 819.24 | 814.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 813.55 | 818.10 | 814.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:15:00 | 813.55 | 818.10 | 814.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 814.70 | 817.42 | 814.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:15:00 | 814.70 | 817.42 | 814.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 822.15 | 824.68 | 820.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 822.15 | 824.68 | 820.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 823.10 | 824.36 | 820.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:15:00 | 823.10 | 824.36 | 820.88 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 820.65 | 823.62 | 820.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:15:00 | 820.65 | 823.62 | 820.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 820.10 | 822.92 | 820.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:15:00 | 820.10 | 822.92 | 820.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 820.00 | 822.33 | 820.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:15:00 | 820.00 | 822.33 | 820.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 826.90 | 823.25 | 821.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 826.90 | 823.25 | 821.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 826.00 | 826.10 | 823.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:15:00 | 826.00 | 826.10 | 823.88 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 824.25 | 825.73 | 823.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 824.25 | 825.73 | 823.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 822.05 | 824.99 | 823.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:15:00 | 822.05 | 824.99 | 823.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 818.25 | 823.64 | 823.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:15:00 | 818.25 | 823.64 | 823.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 828.00 | 824.28 | 823.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 828.00 | 824.28 | 823.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 827.75 | 826.46 | 824.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:15:00 | 827.75 | 826.46 | 824.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 830.10 | 827.19 | 825.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 830.10 | 827.19 | 825.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 830.35 | 827.82 | 825.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:15:00 | 830.35 | 827.82 | 825.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 831.65 | 828.78 | 826.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:15:00 | 831.65 | 828.78 | 826.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 828.60 | 828.74 | 826.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:15:00 | 828.60 | 828.74 | 826.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 827.00 | 828.39 | 826.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 827.00 | 828.39 | 826.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 826.95 | 828.10 | 826.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:15:00 | 826.95 | 828.10 | 826.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 824.10 | 827.30 | 826.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:15:00 | 824.10 | 827.30 | 826.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 830.50 | 827.94 | 826.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:15:00 | 830.50 | 827.94 | 826.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 827.00 | 829.81 | 828.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 827.00 | 829.81 | 828.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 826.50 | 829.15 | 828.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:15:00 | 826.50 | 829.15 | 828.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 837.00 | 830.72 | 829.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:15:00 | 837.00 | 830.72 | 829.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 823.00 | 829.18 | 828.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:15:00 | 823.00 | 829.18 | 828.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 819.00 | 827.14 | 827.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 814.35 | 821.99 | 824.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 818.60 | 817.57 | 821.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 818.60 | 817.57 | 821.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 818.60 | 817.57 | 821.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 818.60 | 817.57 | 821.04 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 817.95 | 817.45 | 820.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:15:00 | 817.95 | 817.45 | 820.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 826.05 | 819.17 | 820.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:15:00 | 826.05 | 819.17 | 820.89 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 831.20 | 821.58 | 821.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:15:00 | 831.20 | 821.58 | 821.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 827.60 | 822.78 | 822.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 13:15:00 | 833.45 | 825.70 | 823.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 821.90 | 827.04 | 825.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 821.90 | 827.04 | 825.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 821.90 | 827.04 | 825.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 821.90 | 827.04 | 825.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 822.50 | 826.13 | 824.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:15:00 | 822.50 | 826.13 | 824.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 831.40 | 827.19 | 825.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:15:00 | 831.40 | 827.19 | 825.54 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 841.90 | 850.43 | 845.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:15:00 | 841.90 | 850.43 | 845.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 843.20 | 848.99 | 845.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:15:00 | 843.20 | 848.99 | 845.30 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 838.00 | 846.79 | 844.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:15:00 | 838.00 | 846.79 | 844.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 831.80 | 841.59 | 842.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 824.95 | 838.26 | 840.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 15:15:00 | 784.50 | 782.25 | 789.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 15:15:00 | 784.50 | 782.25 | 789.81 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 788.60 | 781.96 | 787.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:15:00 | 788.60 | 781.96 | 787.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 782.80 | 782.12 | 786.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:15:00 | 782.80 | 782.12 | 786.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 780.40 | 780.85 | 784.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 780.40 | 780.85 | 784.89 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 776.75 | 769.56 | 774.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:15:00 | 776.75 | 769.56 | 774.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 779.60 | 771.56 | 775.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:15:00 | 779.60 | 771.56 | 775.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 759.90 | 769.23 | 773.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 759.90 | 769.23 | 773.97 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 776.25 | 768.76 | 770.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 776.25 | 768.76 | 770.96 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 776.90 | 770.39 | 771.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:15:00 | 776.90 | 770.39 | 771.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 777.90 | 772.28 | 772.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 789.85 | 775.80 | 773.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 789.40 | 795.39 | 789.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:15:00 | 789.40 | 795.39 | 789.23 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 791.00 | 794.51 | 789.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:15:00 | 791.00 | 794.51 | 789.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 787.25 | 793.06 | 789.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:15:00 | 787.25 | 793.06 | 789.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 789.00 | 792.25 | 789.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:15:00 | 789.00 | 792.25 | 789.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 758.95 | 782.02 | 784.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 753.65 | 772.46 | 779.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 761.65 | 759.75 | 769.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:15:00 | 761.65 | 759.75 | 769.53 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 762.45 | 761.57 | 766.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 762.45 | 761.57 | 766.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 766.60 | 762.58 | 766.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:15:00 | 766.60 | 762.58 | 766.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 757.05 | 761.47 | 765.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:15:00 | 757.05 | 761.47 | 765.51 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 759.55 | 759.69 | 762.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 759.55 | 759.69 | 762.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 743.40 | 752.62 | 757.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:15:00 | 743.40 | 752.62 | 757.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 715.95 | 708.89 | 716.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:15:00 | 715.95 | 708.89 | 716.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 721.10 | 711.33 | 717.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:15:00 | 721.10 | 711.33 | 717.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 721.45 | 713.36 | 717.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:15:00 | 721.45 | 713.36 | 717.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 730.00 | 720.07 | 719.97 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 714.65 | 719.68 | 719.99 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 742.55 | 723.76 | 721.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 747.05 | 728.41 | 724.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 734.80 | 738.07 | 731.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 734.80 | 738.07 | 731.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 734.80 | 738.07 | 731.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 734.80 | 738.07 | 731.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 731.65 | 736.78 | 731.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:15:00 | 731.65 | 736.78 | 731.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 730.30 | 735.49 | 731.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:15:00 | 730.30 | 735.49 | 731.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 733.45 | 735.08 | 731.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:15:00 | 733.45 | 735.08 | 731.91 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 733.40 | 734.74 | 732.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:15:00 | 733.40 | 734.74 | 732.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 725.00 | 732.80 | 731.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:15:00 | 725.00 | 732.80 | 731.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 723.60 | 730.96 | 730.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:15:00 | 723.60 | 730.96 | 730.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 726.80 | 730.13 | 730.34 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 733.50 | 730.80 | 730.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 737.85 | 732.21 | 731.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 737.50 | 743.43 | 738.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 737.50 | 743.43 | 738.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 737.50 | 743.43 | 738.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:15:00 | 737.50 | 743.43 | 738.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 737.45 | 742.24 | 738.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:15:00 | 737.45 | 742.24 | 738.69 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 732.45 | 740.28 | 738.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:15:00 | 732.45 | 740.28 | 738.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 730.55 | 735.65 | 736.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 707.05 | 729.93 | 733.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 725.00 | 713.11 | 720.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 725.00 | 713.11 | 720.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 725.00 | 713.11 | 720.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 725.00 | 713.11 | 720.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 725.30 | 715.55 | 721.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:15:00 | 725.30 | 715.55 | 721.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 727.35 | 719.26 | 721.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:15:00 | 727.35 | 719.26 | 721.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 718.05 | 719.02 | 721.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:15:00 | 718.05 | 719.02 | 721.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 702.85 | 714.82 | 718.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 702.85 | 714.82 | 718.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 668.45 | 646.85 | 662.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 668.45 | 646.85 | 662.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 693.25 | 656.13 | 665.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:15:00 | 693.25 | 656.13 | 665.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 666.80 | 660.90 | 665.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:15:00 | 666.80 | 660.90 | 665.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 660.00 | 660.72 | 665.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:15:00 | 660.00 | 660.72 | 665.04 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 667.35 | 661.87 | 664.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 667.35 | 661.87 | 664.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 661.40 | 661.78 | 664.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:15:00 | 661.40 | 661.78 | 664.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 662.05 | 655.92 | 659.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 662.05 | 655.92 | 659.66 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 673.90 | 659.52 | 660.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:15:00 | 673.90 | 659.52 | 660.96 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 674.20 | 662.46 | 662.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 678.40 | 667.49 | 664.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 652.20 | 668.59 | 666.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 652.20 | 668.59 | 666.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 652.20 | 668.59 | 666.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 652.20 | 668.59 | 666.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 636.95 | 662.26 | 663.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 626.95 | 655.20 | 660.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 648.95 | 644.77 | 652.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 648.95 | 644.77 | 652.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 648.95 | 644.77 | 652.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 648.95 | 644.77 | 652.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 650.30 | 645.88 | 651.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:15:00 | 650.30 | 645.88 | 651.93 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 657.70 | 646.52 | 649.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 657.70 | 646.52 | 649.18 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 655.50 | 648.32 | 649.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:15:00 | 655.50 | 648.32 | 649.75 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 654.00 | 649.86 | 650.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:15:00 | 654.00 | 649.86 | 650.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 663.35 | 652.56 | 651.42 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 650.00 | 654.31 | 654.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 648.90 | 653.51 | 654.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 12:15:00 | 655.95 | 651.90 | 653.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 12:15:00 | 655.95 | 651.90 | 653.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 655.95 | 651.90 | 653.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:15:00 | 655.95 | 651.90 | 653.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 645.85 | 650.69 | 652.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:15:00 | 645.85 | 650.69 | 652.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 640.75 | 636.63 | 643.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:15:00 | 640.75 | 636.63 | 643.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 659.70 | 641.25 | 644.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:15:00 | 659.70 | 641.25 | 644.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 662.55 | 645.51 | 646.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:15:00 | 662.55 | 645.51 | 646.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 09:15:00 | 680.50 | 652.51 | 649.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 10:15:00 | 688.40 | 659.68 | 653.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 12:15:00 | 679.25 | 680.16 | 670.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 12:15:00 | 679.25 | 680.16 | 670.93 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 664.25 | 676.98 | 670.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 13:15:00 | 664.25 | 676.98 | 670.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 665.30 | 674.64 | 669.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:15:00 | 665.30 | 674.64 | 669.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 642.75 | 663.53 | 665.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 635.95 | 658.02 | 662.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 610.35 | 609.84 | 621.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 14:15:00 | 610.35 | 609.84 | 621.95 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 627.55 | 612.61 | 621.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 627.55 | 612.61 | 621.05 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 622.50 | 614.59 | 621.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:15:00 | 622.50 | 614.59 | 621.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 626.20 | 616.91 | 621.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:15:00 | 626.20 | 616.91 | 621.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 622.55 | 618.85 | 621.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:15:00 | 622.55 | 618.85 | 621.75 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 629.00 | 620.88 | 622.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:15:00 | 629.00 | 620.88 | 622.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 628.35 | 622.38 | 622.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:15:00 | 628.35 | 622.38 | 622.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 630.40 | 623.98 | 623.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 640.00 | 627.18 | 625.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 636.20 | 640.66 | 636.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 636.20 | 640.66 | 636.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 636.20 | 640.66 | 636.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:15:00 | 636.20 | 640.66 | 636.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 642.20 | 640.97 | 636.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 642.20 | 640.97 | 636.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 645.85 | 652.57 | 648.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 645.85 | 652.57 | 648.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 642.55 | 650.56 | 647.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:15:00 | 642.55 | 650.56 | 647.84 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 636.80 | 645.77 | 646.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 622.35 | 638.61 | 642.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 606.35 | 604.10 | 614.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:15:00 | 606.35 | 604.10 | 614.65 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 614.90 | 606.26 | 614.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 614.90 | 606.26 | 614.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 623.00 | 609.61 | 615.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:15:00 | 623.00 | 609.61 | 615.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 620.70 | 611.82 | 615.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:15:00 | 620.70 | 611.82 | 615.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 618.00 | 614.69 | 616.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:15:00 | 618.00 | 614.69 | 616.35 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 628.05 | 618.20 | 617.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 634.40 | 621.44 | 619.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 625.00 | 629.42 | 625.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 625.00 | 629.42 | 625.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 625.00 | 629.42 | 625.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:15:00 | 625.00 | 629.42 | 625.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 627.50 | 629.03 | 625.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:15:00 | 627.50 | 629.03 | 625.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 624.35 | 628.10 | 625.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:15:00 | 624.35 | 628.10 | 625.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 638.60 | 630.20 | 626.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:15:00 | 638.60 | 630.20 | 626.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 632.70 | 634.23 | 631.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:15:00 | 632.70 | 634.23 | 631.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 626.40 | 632.67 | 631.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 626.40 | 632.67 | 631.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 624.10 | 630.95 | 630.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:15:00 | 624.10 | 630.95 | 630.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 621.25 | 629.01 | 629.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 603.35 | 621.87 | 626.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 609.10 | 608.27 | 614.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 609.10 | 608.27 | 614.52 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 609.00 | 605.69 | 609.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 609.00 | 605.69 | 609.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 609.20 | 606.39 | 609.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 609.20 | 606.39 | 609.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 612.85 | 607.68 | 609.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:15:00 | 612.85 | 607.68 | 609.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 611.40 | 608.43 | 610.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:15:00 | 611.40 | 608.43 | 610.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 608.75 | 608.49 | 609.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:15:00 | 608.75 | 608.49 | 609.90 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 609.90 | 608.77 | 609.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:15:00 | 609.90 | 608.77 | 609.90 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 607.90 | 608.60 | 609.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:15:00 | 607.90 | 608.60 | 609.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 612.15 | 609.31 | 609.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 612.15 | 609.31 | 609.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 612.45 | 609.94 | 610.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:15:00 | 612.45 | 609.94 | 610.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 612.80 | 610.51 | 610.40 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 605.60 | 609.53 | 609.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 603.90 | 608.02 | 609.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 619.70 | 609.67 | 609.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 619.70 | 609.67 | 609.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 619.70 | 609.67 | 609.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 619.70 | 609.67 | 609.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 621.50 | 612.04 | 610.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 625.15 | 618.76 | 614.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 692.05 | 697.49 | 681.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 692.05 | 697.49 | 681.40 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 682.85 | 693.00 | 682.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:15:00 | 682.85 | 693.00 | 682.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 681.95 | 690.79 | 682.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 681.95 | 690.79 | 682.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 684.10 | 689.45 | 682.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:15:00 | 684.10 | 689.45 | 682.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 680.00 | 687.56 | 682.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:15:00 | 680.00 | 687.56 | 682.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 680.65 | 686.18 | 681.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 680.65 | 686.18 | 681.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 676.10 | 684.16 | 681.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:15:00 | 676.10 | 684.16 | 681.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 672.90 | 681.91 | 680.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:15:00 | 672.90 | 681.91 | 680.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 670.20 | 679.57 | 679.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 661.15 | 675.06 | 677.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 670.85 | 667.79 | 671.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 670.85 | 667.79 | 671.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 670.85 | 667.79 | 671.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 670.85 | 667.79 | 671.12 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 676.95 | 669.62 | 671.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:15:00 | 676.95 | 669.62 | 671.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 675.15 | 670.73 | 671.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:15:00 | 675.15 | 670.73 | 671.96 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 667.80 | 670.31 | 671.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:15:00 | 667.80 | 670.31 | 671.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 671.05 | 670.46 | 671.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:15:00 | 671.05 | 670.46 | 671.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 666.10 | 669.59 | 671.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:15:00 | 666.10 | 669.59 | 671.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 662.85 | 668.24 | 670.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 662.85 | 668.24 | 670.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 644.80 | 656.91 | 662.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 644.80 | 656.91 | 662.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 657.90 | 657.11 | 662.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:15:00 | 657.90 | 657.11 | 662.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 664.45 | 658.58 | 662.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:15:00 | 664.45 | 658.58 | 662.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 667.60 | 660.38 | 663.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:15:00 | 667.60 | 660.38 | 663.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 668.00 | 661.91 | 663.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:15:00 | 668.00 | 661.91 | 663.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 670.00 | 665.12 | 664.75 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 661.45 | 664.39 | 664.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 10:15:00 | 658.60 | 663.23 | 663.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 13:15:00 | 594.70 | 592.91 | 606.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 13:15:00 | 594.70 | 592.91 | 606.97 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 587.30 | 575.54 | 586.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 587.30 | 575.54 | 586.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 593.35 | 579.10 | 587.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:15:00 | 593.35 | 579.10 | 587.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 592.75 | 581.83 | 587.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:15:00 | 592.75 | 581.83 | 587.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 597.70 | 587.82 | 589.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:15:00 | 597.70 | 587.82 | 589.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 593.50 | 588.95 | 589.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:15:00 | 593.50 | 588.95 | 589.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 607.25 | 592.61 | 591.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 618.50 | 606.90 | 599.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 608.20 | 608.58 | 603.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:15:00 | 608.20 | 608.58 | 603.11 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 599.65 | 606.71 | 603.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 599.65 | 606.71 | 603.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 603.30 | 606.02 | 603.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:15:00 | 603.30 | 606.02 | 603.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 604.15 | 605.65 | 603.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:15:00 | 604.15 | 605.65 | 603.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 603.85 | 605.29 | 603.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:15:00 | 603.85 | 605.29 | 603.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 602.55 | 604.74 | 603.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:15:00 | 602.55 | 604.74 | 603.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 607.00 | 605.19 | 604.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:15:00 | 607.00 | 605.19 | 604.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 627.75 | 610.24 | 606.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 627.75 | 610.24 | 606.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 622.00 | 624.30 | 620.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:15:00 | 622.00 | 624.30 | 620.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 620.00 | 623.44 | 620.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 620.00 | 623.44 | 620.23 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 634.30 | 625.61 | 621.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:15:00 | 634.30 | 625.61 | 621.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 622.55 | 625.65 | 622.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:15:00 | 622.55 | 625.65 | 622.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 620.00 | 624.52 | 622.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:15:00 | 620.00 | 624.52 | 622.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 623.90 | 624.40 | 622.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 15:15:00 | 623.90 | 624.40 | 622.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 635.05 | 626.53 | 623.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 635.05 | 626.53 | 623.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 623.85 | 627.44 | 625.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:15:00 | 623.85 | 627.44 | 625.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 624.00 | 626.75 | 625.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:15:00 | 624.00 | 626.75 | 625.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 602.55 | 621.91 | 623.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 597.25 | 616.98 | 620.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 606.25 | 602.87 | 607.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 606.25 | 602.87 | 607.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 606.25 | 602.87 | 607.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 606.25 | 602.87 | 607.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 606.65 | 603.63 | 607.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:15:00 | 606.65 | 603.63 | 607.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 606.60 | 604.22 | 607.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:15:00 | 606.60 | 604.22 | 607.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 604.60 | 604.30 | 607.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:15:00 | 604.60 | 604.30 | 607.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 583.55 | 581.08 | 585.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:15:00 | 583.55 | 581.08 | 585.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 584.50 | 581.76 | 585.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:15:00 | 584.50 | 581.76 | 585.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 584.70 | 582.35 | 585.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:15:00 | 584.70 | 582.35 | 585.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 586.65 | 583.21 | 585.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:15:00 | 586.65 | 583.21 | 585.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 589.75 | 584.52 | 586.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:15:00 | 589.75 | 584.52 | 586.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 583.65 | 584.34 | 585.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 583.65 | 584.34 | 585.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 575.10 | 566.77 | 571.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:15:00 | 575.10 | 566.77 | 571.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 571.50 | 567.71 | 571.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 571.50 | 567.71 | 571.35 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 573.80 | 569.34 | 571.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:15:00 | 573.80 | 569.34 | 571.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 568.00 | 569.08 | 571.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:15:00 | 568.00 | 569.08 | 571.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 564.05 | 568.07 | 570.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:15:00 | 564.05 | 568.07 | 570.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 600.75 | 562.49 | 562.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 600.75 | 562.49 | 562.81 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 595.85 | 569.17 | 565.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 602.10 | 575.75 | 569.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 659.85 | 661.88 | 651.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 659.85 | 661.88 | 651.85 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 654.10 | 660.60 | 654.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:15:00 | 654.10 | 660.60 | 654.56 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 654.10 | 659.30 | 654.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:15:00 | 654.10 | 659.30 | 654.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 656.90 | 658.82 | 654.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 656.90 | 658.82 | 654.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 669.35 | 660.92 | 656.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 669.35 | 660.92 | 656.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 670.00 | 666.12 | 661.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 670.00 | 666.12 | 661.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 670.50 | 669.10 | 665.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 670.50 | 669.10 | 665.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 667.05 | 669.96 | 667.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:15:00 | 667.05 | 669.96 | 667.69 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 667.00 | 669.37 | 667.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:15:00 | 667.00 | 669.37 | 667.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 719.10 | 684.84 | 676.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 719.10 | 684.84 | 676.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 750.60 | 753.31 | 745.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:15:00 | 750.60 | 753.31 | 745.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 761.55 | 754.43 | 747.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 761.55 | 754.43 | 747.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 763.15 | 760.87 | 757.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 763.15 | 760.87 | 757.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 765.00 | 768.42 | 765.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:15:00 | 765.00 | 768.42 | 765.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 766.90 | 768.11 | 765.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 766.90 | 768.11 | 765.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 781.40 | 770.77 | 767.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 781.40 | 770.77 | 767.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 786.90 | 790.25 | 784.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:15:00 | 786.90 | 790.25 | 784.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 787.45 | 789.69 | 784.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 787.45 | 789.69 | 784.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 780.95 | 787.94 | 784.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:15:00 | 780.95 | 787.94 | 784.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 779.00 | 786.15 | 783.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:15:00 | 779.00 | 786.15 | 783.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 771.05 | 781.59 | 782.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 763.30 | 775.24 | 778.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 730.35 | 729.67 | 742.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:15:00 | 730.35 | 729.67 | 742.77 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 739.50 | 731.63 | 742.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 739.50 | 731.63 | 742.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 753.50 | 737.44 | 742.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:15:00 | 753.50 | 737.44 | 742.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 749.00 | 739.76 | 743.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:15:00 | 749.00 | 739.76 | 743.14 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 769.90 | 745.78 | 745.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 786.00 | 771.97 | 762.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 764.50 | 774.79 | 768.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:15:00 | 764.50 | 774.79 | 768.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 765.60 | 772.96 | 768.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:15:00 | 765.60 | 772.96 | 768.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 764.90 | 771.34 | 767.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:15:00 | 764.90 | 771.34 | 767.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 760.00 | 769.08 | 767.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:15:00 | 760.00 | 769.08 | 767.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 751.00 | 763.13 | 764.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 749.65 | 759.61 | 762.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 762.00 | 760.15 | 761.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:15:00 | 762.00 | 760.15 | 761.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 764.40 | 761.00 | 761.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:15:00 | 764.40 | 761.00 | 761.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 768.90 | 762.58 | 762.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 773.00 | 765.34 | 763.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 798.00 | 798.83 | 791.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:15:00 | 798.00 | 798.83 | 791.22 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 796.45 | 800.36 | 795.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 796.45 | 800.36 | 795.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 802.00 | 800.69 | 796.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:15:00 | 802.00 | 800.69 | 796.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 832.40 | 807.03 | 799.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 832.40 | 807.03 | 799.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 826.00 | 832.52 | 825.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 826.00 | 832.52 | 825.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 824.95 | 831.01 | 825.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:15:00 | 824.95 | 831.01 | 825.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 826.50 | 830.10 | 825.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:15:00 | 826.50 | 830.10 | 825.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 826.20 | 829.32 | 825.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:15:00 | 826.20 | 829.32 | 825.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 826.75 | 828.81 | 825.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:15:00 | 826.75 | 828.81 | 825.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 825.00 | 828.05 | 825.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:15:00 | 825.00 | 828.05 | 825.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 822.50 | 826.94 | 825.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:15:00 | 822.50 | 826.94 | 825.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 825.00 | 826.72 | 825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:15:00 | 825.00 | 826.72 | 825.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 828.90 | 827.16 | 825.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:15:00 | 828.90 | 827.16 | 825.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 827.00 | 827.12 | 825.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:15:00 | 827.00 | 827.12 | 825.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 821.30 | 825.96 | 825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:15:00 | 821.30 | 825.96 | 825.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 820.35 | 824.84 | 825.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 817.80 | 822.64 | 823.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 817.05 | 820.25 | 821.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:15:00 | 817.05 | 820.25 | 821.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 822.95 | 820.79 | 821.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:15:00 | 822.95 | 820.79 | 821.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 817.95 | 820.22 | 821.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:15:00 | 817.95 | 820.22 | 821.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 824.00 | 819.39 | 820.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:15:00 | 824.00 | 819.39 | 820.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 816.05 | 818.72 | 820.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 816.05 | 818.72 | 820.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 820.00 | 818.98 | 820.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:15:00 | 820.00 | 818.98 | 820.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 816.10 | 818.40 | 819.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 816.10 | 818.40 | 819.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 814.95 | 817.71 | 819.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:15:00 | 814.95 | 817.71 | 819.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 819.00 | 817.06 | 818.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:15:00 | 819.00 | 817.06 | 818.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 814.85 | 816.61 | 818.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:15:00 | 814.85 | 816.61 | 818.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 815.30 | 816.35 | 818.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 815.30 | 816.35 | 818.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 826.00 | 818.28 | 818.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:15:00 | 826.00 | 818.28 | 818.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 826.00 | 819.83 | 819.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 829.95 | 824.52 | 822.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 840.00 | 844.90 | 840.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:15:00 | 840.00 | 844.90 | 840.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 841.00 | 844.12 | 840.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:15:00 | 841.00 | 844.12 | 840.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 843.95 | 844.08 | 840.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:15:00 | 843.95 | 844.08 | 840.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 848.00 | 844.87 | 841.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 848.00 | 844.87 | 841.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 903.60 | 898.21 | 884.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 903.60 | 898.21 | 884.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 923.00 | 914.12 | 906.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 923.00 | 914.12 | 906.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 927.95 | 933.22 | 925.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 927.95 | 933.22 | 925.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 916.00 | 929.78 | 924.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:15:00 | 916.00 | 929.78 | 924.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 923.45 | 928.51 | 924.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:15:00 | 923.45 | 928.51 | 924.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 926.00 | 928.01 | 924.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:15:00 | 926.00 | 928.01 | 924.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 920.65 | 926.54 | 924.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:15:00 | 920.65 | 926.54 | 924.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 930.00 | 927.23 | 924.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:15:00 | 930.00 | 927.23 | 924.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 913.75 | 924.20 | 923.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 913.75 | 924.20 | 923.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 908.70 | 921.10 | 922.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 882.00 | 907.34 | 914.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 898.90 | 898.52 | 907.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:15:00 | 898.90 | 898.52 | 907.51 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 901.00 | 894.56 | 900.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:15:00 | 901.00 | 894.56 | 900.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 909.95 | 897.64 | 901.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:15:00 | 909.95 | 897.64 | 901.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 909.70 | 900.05 | 902.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:15:00 | 909.70 | 900.05 | 902.28 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 925.50 | 905.14 | 904.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 941.25 | 916.18 | 909.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 998.00 | 1002.99 | 980.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:15:00 | 998.00 | 1002.99 | 980.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 997.00 | 1001.47 | 983.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 997.00 | 1001.47 | 983.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1012.00 | 1003.58 | 986.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1012.00 | 1003.58 | 986.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1013.50 | 1023.83 | 1015.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 1013.50 | 1023.83 | 1015.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1009.30 | 1020.92 | 1014.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:15:00 | 1009.30 | 1020.92 | 1014.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1006.90 | 1018.12 | 1013.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:15:00 | 1006.90 | 1018.12 | 1013.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1000.00 | 1014.49 | 1012.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:15:00 | 1000.00 | 1014.49 | 1012.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 986.30 | 1006.78 | 1009.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 951.50 | 985.20 | 997.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 882.60 | 869.60 | 885.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:15:00 | 882.60 | 869.60 | 885.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 886.30 | 872.94 | 885.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:15:00 | 886.30 | 872.94 | 885.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 886.30 | 875.62 | 885.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:15:00 | 886.30 | 875.62 | 885.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 869.10 | 868.82 | 875.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 869.10 | 868.82 | 875.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 870.20 | 857.52 | 865.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:15:00 | 870.20 | 857.52 | 865.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 867.30 | 859.47 | 866.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:15:00 | 867.30 | 859.47 | 866.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 859.80 | 859.54 | 865.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 859.80 | 859.54 | 865.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 857.00 | 853.69 | 858.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 857.00 | 853.69 | 858.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 845.00 | 851.95 | 857.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 845.00 | 851.95 | 857.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 854.00 | 847.49 | 852.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 854.00 | 847.49 | 852.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 862.00 | 850.39 | 853.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:15:00 | 862.00 | 850.39 | 853.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 860.10 | 852.33 | 854.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:15:00 | 860.10 | 852.33 | 854.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 860.00 | 855.59 | 855.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 863.10 | 857.51 | 856.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 13:15:00 | 860.20 | 860.79 | 858.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:15:00 | 860.20 | 860.79 | 858.52 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 860.00 | 860.63 | 858.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:15:00 | 860.00 | 860.63 | 858.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 861.90 | 860.89 | 858.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:15:00 | 861.90 | 860.89 | 858.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 860.00 | 860.71 | 859.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 860.00 | 860.71 | 859.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 857.80 | 860.13 | 858.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:15:00 | 857.80 | 860.13 | 858.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 860.00 | 860.10 | 859.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:15:00 | 860.00 | 860.10 | 859.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 867.50 | 861.58 | 859.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 867.50 | 861.58 | 859.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 860.00 | 861.22 | 859.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:15:00 | 860.00 | 861.22 | 859.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 858.00 | 860.58 | 859.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:15:00 | 858.00 | 860.58 | 859.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 861.80 | 860.82 | 859.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 861.80 | 860.82 | 859.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 860.00 | 860.66 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:15:00 | 860.00 | 860.66 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 860.00 | 860.53 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 860.00 | 860.53 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 860.00 | 860.42 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:15:00 | 860.00 | 860.42 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 860.00 | 860.34 | 859.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:15:00 | 860.00 | 860.34 | 859.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 860.00 | 860.17 | 859.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 860.00 | 860.17 | 859.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 866.50 | 861.43 | 860.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 866.50 | 861.43 | 860.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 863.50 | 862.41 | 861.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:15:00 | 863.50 | 862.41 | 861.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 864.00 | 862.73 | 861.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:15:00 | 864.00 | 862.73 | 861.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 865.00 | 863.18 | 861.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:15:00 | 865.00 | 863.18 | 861.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 855.50 | 861.65 | 861.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 855.50 | 861.65 | 861.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 858.70 | 861.06 | 860.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:15:00 | 858.70 | 861.06 | 860.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 856.00 | 860.05 | 860.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 09:15:00 | 854.00 | 858.16 | 859.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 873.80 | 855.75 | 855.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:15:00 | 873.80 | 855.75 | 855.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 875.20 | 859.64 | 857.67 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 858.00 | 863.22 | 863.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 850.00 | 858.94 | 861.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 880.05 | 864.62 | 863.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 11:15:00 | 886.00 | 868.90 | 865.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 880.00 | 886.69 | 879.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:15:00 | 880.00 | 886.69 | 879.52 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 877.95 | 884.94 | 879.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:15:00 | 877.95 | 884.94 | 879.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 874.85 | 882.92 | 878.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:15:00 | 874.85 | 882.92 | 878.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 860.00 | 875.45 | 876.08 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 881.45 | 874.35 | 874.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 886.70 | 876.82 | 875.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 877.45 | 879.55 | 877.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:15:00 | 877.45 | 879.55 | 877.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 876.75 | 878.99 | 877.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:15:00 | 876.75 | 878.99 | 877.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 874.00 | 877.99 | 877.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:15:00 | 874.00 | 877.99 | 877.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 866.80 | 875.76 | 876.22 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 887.05 | 877.58 | 876.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 903.70 | 889.21 | 883.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 902.60 | 907.72 | 901.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:15:00 | 902.60 | 907.72 | 901.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 899.10 | 906.00 | 901.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:15:00 | 899.10 | 906.00 | 901.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 895.00 | 903.80 | 900.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:15:00 | 895.00 | 903.80 | 900.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 901.35 | 903.31 | 901.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:15:00 | 901.35 | 903.31 | 901.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 901.50 | 902.95 | 901.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 901.50 | 902.95 | 901.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 900.45 | 902.45 | 901.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 900.45 | 902.45 | 901.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 898.60 | 901.68 | 900.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 898.60 | 901.68 | 900.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 893.35 | 900.01 | 900.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 890.00 | 896.41 | 898.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 844.95 | 839.94 | 846.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:15:00 | 844.95 | 839.94 | 846.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 832.15 | 823.87 | 830.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:15:00 | 832.15 | 823.87 | 830.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 829.00 | 824.90 | 830.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:15:00 | 829.00 | 824.90 | 830.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 827.95 | 825.80 | 830.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:15:00 | 827.95 | 825.80 | 830.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 844.05 | 829.45 | 831.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 844.05 | 829.45 | 831.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 850.80 | 833.72 | 833.11 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 838.00 | 841.80 | 842.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 832.30 | 839.90 | 841.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 833.40 | 836.17 | 838.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:15:00 | 833.40 | 836.17 | 838.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 824.95 | 833.55 | 837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 824.95 | 833.55 | 837.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 827.45 | 828.02 | 832.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:15:00 | 827.45 | 828.02 | 832.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 836.55 | 829.28 | 832.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 836.55 | 829.28 | 832.33 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 842.95 | 832.01 | 833.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:15:00 | 842.95 | 832.01 | 833.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 831.50 | 831.91 | 833.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:15:00 | 831.50 | 831.91 | 833.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 826.40 | 829.10 | 831.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 826.40 | 829.10 | 831.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 833.25 | 829.48 | 830.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 833.25 | 829.48 | 830.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 833.00 | 830.19 | 830.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:15:00 | 833.00 | 830.19 | 830.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 836.85 | 832.29 | 831.80 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 826.30 | 831.25 | 831.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 817.35 | 828.47 | 830.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 826.50 | 823.37 | 825.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:15:00 | 826.50 | 823.37 | 825.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 825.15 | 823.72 | 825.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:15:00 | 825.15 | 823.72 | 825.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 822.40 | 823.46 | 825.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:15:00 | 822.40 | 823.46 | 825.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 829.05 | 824.42 | 825.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:15:00 | 829.05 | 824.42 | 825.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 829.75 | 825.49 | 825.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:15:00 | 829.75 | 825.49 | 825.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 832.65 | 826.92 | 826.55 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 823.30 | 826.55 | 826.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 813.00 | 823.33 | 825.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 820.40 | 819.46 | 821.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:15:00 | 820.40 | 819.46 | 821.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 827.40 | 821.05 | 822.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 827.40 | 821.05 | 822.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 816.85 | 821.42 | 822.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:15:00 | 816.85 | 821.42 | 822.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 830.00 | 819.45 | 820.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 830.00 | 819.45 | 820.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 840.50 | 823.66 | 822.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 859.55 | 839.37 | 831.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 850.10 | 859.71 | 854.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 850.10 | 859.71 | 854.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 852.00 | 858.17 | 854.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 852.00 | 858.17 | 854.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 857.55 | 857.21 | 854.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:15:00 | 857.55 | 857.21 | 854.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 854.15 | 856.60 | 854.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:15:00 | 854.15 | 856.60 | 854.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 853.15 | 855.91 | 854.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:15:00 | 853.15 | 855.91 | 854.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 868.40 | 858.41 | 855.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 868.40 | 858.41 | 855.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 857.00 | 858.06 | 856.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:15:00 | 857.00 | 858.06 | 856.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 851.85 | 856.82 | 855.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:15:00 | 851.85 | 856.82 | 855.65 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 852.55 | 855.96 | 855.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 852.55 | 855.96 | 855.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 851.40 | 854.45 | 854.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 841.70 | 851.90 | 853.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 857.00 | 846.32 | 848.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:15:00 | 857.00 | 846.32 | 848.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 856.90 | 848.44 | 849.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:15:00 | 856.90 | 848.44 | 849.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 856.40 | 850.03 | 850.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 862.85 | 853.50 | 851.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 873.00 | 882.34 | 872.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 873.00 | 882.34 | 872.41 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 874.70 | 880.81 | 872.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:15:00 | 874.70 | 880.81 | 872.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 872.45 | 879.14 | 872.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:15:00 | 872.45 | 879.14 | 872.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 875.10 | 878.33 | 872.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:15:00 | 875.10 | 878.33 | 872.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 875.65 | 877.79 | 873.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:15:00 | 875.65 | 877.79 | 873.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 876.00 | 877.41 | 873.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:15:00 | 876.00 | 877.41 | 873.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 858.85 | 873.70 | 872.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 858.85 | 873.70 | 872.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 845.35 | 868.03 | 869.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 808.00 | 850.81 | 860.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 821.30 | 810.79 | 822.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:15:00 | 821.30 | 810.79 | 822.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 820.15 | 813.66 | 821.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:15:00 | 820.15 | 813.66 | 821.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 817.50 | 814.43 | 821.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:15:00 | 817.50 | 814.43 | 821.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 820.00 | 815.54 | 821.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:15:00 | 820.00 | 815.54 | 821.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 819.90 | 816.42 | 821.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:15:00 | 819.90 | 816.42 | 821.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 823.00 | 817.73 | 821.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:15:00 | 823.00 | 817.73 | 821.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 821.00 | 818.39 | 821.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:15:00 | 821.00 | 818.39 | 821.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 808.65 | 816.44 | 820.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 808.65 | 816.44 | 820.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 781.05 | 771.26 | 778.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 781.05 | 771.26 | 778.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 779.00 | 772.80 | 778.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:15:00 | 779.00 | 772.80 | 778.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 781.00 | 774.44 | 778.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:15:00 | 781.00 | 774.44 | 778.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 782.20 | 775.99 | 779.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:15:00 | 782.20 | 775.99 | 779.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 729.45 | 722.11 | 731.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 729.45 | 722.11 | 731.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 727.40 | 723.17 | 731.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 727.40 | 723.17 | 731.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 730.30 | 725.46 | 730.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:15:00 | 730.30 | 725.46 | 730.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 733.75 | 727.12 | 730.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:15:00 | 733.75 | 727.12 | 730.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 733.20 | 728.34 | 730.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:15:00 | 733.20 | 728.34 | 730.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 722.60 | 720.84 | 724.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:15:00 | 722.60 | 720.84 | 724.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 717.20 | 720.11 | 723.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:15:00 | 717.20 | 720.11 | 723.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 723.00 | 718.99 | 721.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 723.00 | 718.99 | 721.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 723.70 | 719.93 | 721.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:15:00 | 723.70 | 719.93 | 721.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 719.10 | 719.77 | 721.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 719.10 | 719.77 | 721.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 720.90 | 719.19 | 720.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:15:00 | 720.90 | 719.19 | 720.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 722.50 | 719.85 | 720.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 722.50 | 719.85 | 720.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 722.45 | 720.37 | 720.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:15:00 | 722.45 | 720.37 | 720.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 735.65 | 723.43 | 722.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 13:15:00 | 740.30 | 727.85 | 724.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 10:15:00 | 759.75 | 762.55 | 750.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:15:00 | 759.75 | 762.55 | 750.96 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 756.00 | 760.45 | 752.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:15:00 | 756.00 | 760.45 | 752.82 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 764.95 | 761.35 | 753.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:15:00 | 764.95 | 761.35 | 753.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 759.50 | 761.55 | 755.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 759.50 | 761.55 | 755.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 750.70 | 761.09 | 757.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:15:00 | 750.70 | 761.09 | 757.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 753.25 | 759.52 | 756.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:15:00 | 753.25 | 759.52 | 756.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 750.50 | 758.58 | 757.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 750.50 | 758.58 | 757.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 750.95 | 757.06 | 756.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:15:00 | 750.95 | 757.06 | 756.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 751.80 | 756.01 | 756.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 749.00 | 754.60 | 755.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 742.25 | 742.17 | 747.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:15:00 | 742.25 | 742.17 | 747.37 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 750.75 | 743.76 | 747.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:15:00 | 750.75 | 743.76 | 747.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 747.55 | 744.52 | 747.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 747.55 | 744.52 | 747.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 741.35 | 743.89 | 746.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 741.35 | 743.89 | 746.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 719.00 | 717.32 | 720.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 719.00 | 717.32 | 720.93 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 717.50 | 717.35 | 720.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:15:00 | 717.50 | 717.35 | 720.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 710.45 | 710.52 | 713.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 710.45 | 710.52 | 713.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 693.75 | 697.55 | 704.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 693.75 | 697.55 | 704.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 700.45 | 697.62 | 701.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:15:00 | 700.45 | 697.62 | 701.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 710.15 | 700.47 | 702.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 710.15 | 700.47 | 702.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 710.00 | 702.37 | 702.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:15:00 | 710.00 | 702.37 | 702.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 709.35 | 703.77 | 703.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 714.10 | 705.83 | 704.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 736.60 | 737.97 | 728.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:15:00 | 736.60 | 737.97 | 728.54 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 738.00 | 743.14 | 738.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:15:00 | 738.00 | 743.14 | 738.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 739.00 | 742.31 | 738.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:15:00 | 739.00 | 742.31 | 738.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 734.95 | 740.84 | 738.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 734.95 | 740.84 | 738.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 732.05 | 739.08 | 737.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 732.05 | 739.08 | 737.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 732.70 | 736.54 | 736.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 727.10 | 734.65 | 735.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 711.20 | 708.60 | 715.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 711.20 | 708.60 | 715.95 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 722.00 | 711.28 | 716.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 722.00 | 711.28 | 716.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 721.85 | 713.40 | 716.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 721.85 | 713.40 | 716.98 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 724.20 | 718.10 | 718.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:15:00 | 724.20 | 718.10 | 718.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 722.90 | 719.06 | 718.88 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 715.75 | 718.36 | 718.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 712.35 | 716.38 | 717.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 721.30 | 717.03 | 717.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 721.30 | 717.03 | 717.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 715.05 | 716.64 | 717.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:15:00 | 715.05 | 716.64 | 717.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 719.50 | 717.21 | 717.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:15:00 | 719.50 | 717.21 | 717.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 716.80 | 717.13 | 717.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:15:00 | 716.80 | 717.13 | 717.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 716.95 | 717.09 | 717.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:15:00 | 716.95 | 717.09 | 717.33 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 714.50 | 716.57 | 717.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:15:00 | 714.50 | 716.57 | 717.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 703.00 | 705.63 | 710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 703.00 | 705.63 | 710.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 696.20 | 698.35 | 703.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 696.20 | 698.35 | 703.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 701.60 | 698.43 | 702.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:15:00 | 701.60 | 698.43 | 702.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 703.90 | 699.52 | 702.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 703.90 | 699.52 | 702.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 712.35 | 702.09 | 703.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 712.35 | 702.09 | 703.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 710.00 | 703.67 | 703.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:15:00 | 710.00 | 703.67 | 703.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 699.20 | 702.79 | 703.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:15:00 | 699.20 | 702.79 | 703.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 627.00 | 622.62 | 627.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 627.00 | 622.62 | 627.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 622.20 | 622.54 | 626.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:15:00 | 622.20 | 622.54 | 626.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 611.35 | 588.87 | 596.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 611.35 | 588.87 | 596.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 633.60 | 597.82 | 599.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:15:00 | 633.60 | 597.82 | 599.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 623.10 | 602.87 | 601.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 661.65 | 634.68 | 624.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 678.45 | 678.57 | 657.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:15:00 | 678.45 | 678.57 | 657.99 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 680.90 | 689.88 | 681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 680.90 | 689.88 | 681.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 685.00 | 688.90 | 682.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:15:00 | 685.00 | 688.90 | 682.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 684.40 | 688.00 | 682.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:15:00 | 684.40 | 688.00 | 682.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 741.40 | 739.53 | 728.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:15:00 | 741.40 | 739.53 | 728.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 732.85 | 738.69 | 730.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:15:00 | 732.85 | 738.69 | 730.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 765.00 | 742.94 | 733.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 765.00 | 742.94 | 733.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 790.00 | 798.96 | 790.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:15:00 | 790.00 | 798.96 | 790.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 790.00 | 797.17 | 790.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:15:00 | 790.00 | 797.17 | 790.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 814.25 | 800.73 | 793.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 814.25 | 800.73 | 793.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 798.40 | 805.69 | 800.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:15:00 | 798.40 | 805.69 | 800.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 814.50 | 807.46 | 801.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:15:00 | 814.50 | 807.46 | 801.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 899.00 | 915.41 | 905.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 899.00 | 915.41 | 905.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 908.60 | 914.05 | 905.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:15:00 | 908.60 | 914.05 | 905.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 914.30 | 914.10 | 906.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:15:00 | 914.30 | 914.10 | 906.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 914.90 | 913.51 | 907.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:15:00 | 914.90 | 913.51 | 907.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 870.00 | 904.73 | 904.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 870.00 | 904.73 | 904.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 885.00 | 900.78 | 902.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 868.75 | 894.38 | 899.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 892.05 | 888.58 | 894.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:15:00 | 892.05 | 888.58 | 894.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 872.90 | 880.73 | 887.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:15:00 | 872.90 | 880.73 | 887.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 887.80 | 879.71 | 885.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 887.80 | 879.71 | 885.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 888.00 | 881.37 | 885.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 888.00 | 881.37 | 885.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 885.00 | 882.10 | 885.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:15:00 | 885.00 | 882.10 | 885.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 893.00 | 884.28 | 886.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:15:00 | 893.00 | 884.28 | 886.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 891.00 | 885.62 | 886.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:15:00 | 891.00 | 885.62 | 886.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 899.60 | 888.42 | 887.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 900.25 | 890.78 | 888.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 890.00 | 895.18 | 893.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:15:00 | 890.00 | 895.18 | 893.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 875.00 | 891.14 | 891.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 863.10 | 885.53 | 889.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 874.35 | 857.18 | 865.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:15:00 | 874.35 | 857.18 | 865.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 874.35 | 868.72 | 868.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 909.00 | 876.78 | 872.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 897.40 | 892.11 | 887.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:15:00 | 897.40 | 892.11 | 887.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 900.00 | 893.69 | 888.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:15:00 | 900.00 | 893.69 | 888.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 908.95 | 915.42 | 906.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:15:00 | 908.95 | 915.42 | 906.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 899.00 | 912.14 | 906.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:15:00 | 899.00 | 912.14 | 906.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 898.95 | 909.50 | 905.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:15:00 | 898.95 | 909.50 | 905.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 895.00 | 905.24 | 904.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 895.00 | 905.24 | 904.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 901.75 | 904.97 | 904.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:15:00 | 901.75 | 904.97 | 904.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 899.95 | 903.97 | 904.18 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 905.00 | 904.34 | 904.32 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 882.20 | 900.41 | 902.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 862.90 | 888.81 | 896.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 874.00 | 868.45 | 879.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:15:00 | 874.00 | 868.45 | 879.81 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 888.00 | 873.41 | 880.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 888.00 | 873.41 | 880.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 882.00 | 875.13 | 880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 882.00 | 875.13 | 880.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 882.80 | 876.66 | 880.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:15:00 | 882.80 | 876.66 | 880.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 876.35 | 876.60 | 880.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:15:00 | 876.35 | 876.60 | 880.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 875.00 | 876.28 | 879.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:15:00 | 875.00 | 876.28 | 879.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 892.50 | 879.52 | 880.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:15:00 | 892.50 | 879.52 | 880.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 889.10 | 881.44 | 881.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:15:00 | 889.10 | 881.44 | 881.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 905.10 | 886.17 | 883.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 910.00 | 899.05 | 894.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 900.00 | 900.54 | 897.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:15:00 | 900.00 | 900.54 | 897.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 892.95 | 899.02 | 896.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:15:00 | 892.95 | 899.02 | 896.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 890.00 | 897.21 | 896.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:15:00 | 890.00 | 897.21 | 896.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 851.45 | 888.06 | 892.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 846.10 | 874.21 | 884.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 878.40 | 872.14 | 881.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:15:00 | 878.40 | 872.14 | 881.79 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 870.00 | 871.71 | 880.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 14:15:00 | 870.00 | 871.71 | 880.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 875.00 | 870.49 | 878.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 875.00 | 870.49 | 878.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 876.45 | 871.20 | 877.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:15:00 | 876.45 | 871.20 | 877.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 875.00 | 871.96 | 877.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:15:00 | 875.00 | 871.96 | 877.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 870.20 | 871.61 | 876.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:15:00 | 870.20 | 871.61 | 876.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 897.55 | 876.80 | 878.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:15:00 | 897.55 | 876.80 | 878.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 895.35 | 880.51 | 879.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 906.70 | 885.74 | 882.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 892.50 | 894.59 | 888.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:15:00 | 892.50 | 894.59 | 888.93 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 884.10 | 892.24 | 888.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 884.10 | 892.24 | 888.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 879.90 | 889.77 | 888.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 879.90 | 889.77 | 888.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 876.00 | 887.02 | 886.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 876.00 | 887.02 | 886.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 873.00 | 884.21 | 885.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 868.50 | 881.07 | 884.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 870.00 | 869.55 | 870.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:15:00 | 870.00 | 869.55 | 870.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 871.00 | 869.84 | 870.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:15:00 | 871.00 | 869.84 | 870.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 882.80 | 872.43 | 871.41 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 850.00 | 867.95 | 869.46 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 895.95 | 871.76 | 870.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 907.90 | 878.99 | 873.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 923.00 | 928.09 | 914.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:15:00 | 923.00 | 928.09 | 914.85 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1003.00 | 996.34 | 979.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1003.00 | 996.34 | 979.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 1008.00 | 1014.41 | 1002.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:15:00 | 1008.00 | 1014.41 | 1002.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 997.00 | 1010.22 | 1002.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:15:00 | 997.00 | 1010.22 | 1002.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1000.00 | 1008.18 | 1002.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:15:00 | 1000.00 | 1008.18 | 1002.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 999.15 | 1003.76 | 1001.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:15:00 | 999.15 | 1003.76 | 1001.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 983.10 | 999.63 | 999.63 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1038.50 | 1004.79 | 1001.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1039.50 | 1011.73 | 1004.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 13:15:00 | 1075.00 | 1077.45 | 1054.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 13:15:00 | 1075.00 | 1077.45 | 1054.42 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1060.00 | 1072.33 | 1055.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1060.00 | 1072.33 | 1055.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1075.00 | 1072.87 | 1057.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1075.00 | 1072.87 | 1057.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1105.00 | 1121.43 | 1106.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1105.00 | 1121.43 | 1106.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1121.35 | 1122.47 | 1109.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:15:00 | 1121.35 | 1122.47 | 1109.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1170.00 | 1174.43 | 1155.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 1170.00 | 1174.43 | 1155.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1160.00 | 1170.83 | 1157.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:15:00 | 1160.00 | 1170.83 | 1157.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1152.90 | 1167.25 | 1156.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:15:00 | 1152.90 | 1167.25 | 1156.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1164.85 | 1166.77 | 1157.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:15:00 | 1164.85 | 1166.77 | 1157.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1138.50 | 1161.11 | 1155.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:15:00 | 1138.50 | 1161.11 | 1155.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1162.50 | 1161.39 | 1156.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 1162.50 | 1161.39 | 1156.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1149.95 | 1159.10 | 1155.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1149.95 | 1159.10 | 1155.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1165.00 | 1160.28 | 1156.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 1165.00 | 1160.28 | 1156.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1182.00 | 1174.16 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:15:00 | 1182.00 | 1174.16 | 1165.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1169.00 | 1175.50 | 1167.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1169.00 | 1175.50 | 1167.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1200.00 | 1180.40 | 1170.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 1200.00 | 1180.40 | 1170.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1189.00 | 1182.12 | 1172.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:15:00 | 1189.00 | 1182.12 | 1172.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1288.90 | 1324.90 | 1294.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1288.90 | 1324.90 | 1294.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1300.00 | 1319.92 | 1295.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1300.00 | 1319.92 | 1295.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1294.50 | 1314.83 | 1295.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 1294.50 | 1314.83 | 1295.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1295.00 | 1310.87 | 1295.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:15:00 | 1295.00 | 1310.87 | 1295.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1283.00 | 1305.29 | 1294.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:15:00 | 1283.00 | 1305.29 | 1294.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1282.50 | 1300.74 | 1293.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 1282.50 | 1300.74 | 1293.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1282.50 | 1297.09 | 1292.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:15:00 | 1282.50 | 1297.09 | 1292.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1293.00 | 1296.58 | 1292.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:15:00 | 1293.00 | 1296.58 | 1292.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1313.00 | 1299.86 | 1294.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:15:00 | 1313.00 | 1299.86 | 1294.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1345.00 | 1327.92 | 1312.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 1345.00 | 1327.92 | 1312.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1345.00 | 1339.13 | 1324.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:15:00 | 1345.00 | 1339.13 | 1324.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 1309.00 | 1334.52 | 1325.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:15:00 | 1309.00 | 1334.52 | 1325.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 10:15:00 | 1300.00 | 1327.62 | 1322.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 10:15:00 | 1300.00 | 1327.62 | 1322.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-05-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 13:15:00 | 1288.50 | 1314.10 | 1317.32 | EMA200 below EMA400 |

