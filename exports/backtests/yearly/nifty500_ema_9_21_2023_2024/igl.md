# Indraprastha Gas Ltd. (IGL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 165.97
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 158 |
| ALERT2 | 151 |
| ALERT2_SKIP | 83 |
| ALERT3 | 407 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 195 |
| PARTIAL | 18 |
| TARGET_HIT | 1 |
| STOP_HIT | 202 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 221 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 73 / 148
- **Target hits / Stop hits / Partials:** 1 / 202 / 18
- **Avg / median % per leg:** -0.04% / -0.66%
- **Sum % (uncompounded):** -9.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 96 | 21 | 21.9% | 1 | 95 | 0 | -0.70% | -67.1% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.51% | -7.6% |
| BUY @ 3rd Alert (retest2) | 91 | 21 | 23.1% | 1 | 90 | 0 | -0.65% | -59.5% |
| SELL (all) | 125 | 52 | 41.6% | 0 | 107 | 18 | 0.46% | 57.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.16% | -0.5% |
| SELL @ 3rd Alert (retest2) | 122 | 52 | 42.6% | 0 | 104 | 18 | 0.48% | 58.1% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.01% | -8.0% |
| retest2 (combined) | 213 | 73 | 34.3% | 1 | 194 | 18 | -0.01% | -1.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 240.88 | 238.98 | 238.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 09:15:00 | 243.30 | 239.92 | 239.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 12:15:00 | 240.00 | 240.35 | 239.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 12:15:00 | 240.00 | 240.35 | 239.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 240.00 | 240.35 | 239.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:00:00 | 240.00 | 240.35 | 239.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 239.95 | 240.27 | 239.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:45:00 | 239.98 | 240.27 | 239.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 239.98 | 240.22 | 239.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:45:00 | 239.50 | 240.22 | 239.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 239.75 | 240.12 | 239.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:15:00 | 239.28 | 240.12 | 239.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 242.13 | 240.52 | 240.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 10:15:00 | 242.78 | 240.52 | 240.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 14:15:00 | 239.80 | 240.37 | 240.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 239.80 | 240.37 | 240.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 15:15:00 | 239.25 | 240.14 | 240.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 10:15:00 | 240.63 | 240.12 | 240.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 10:15:00 | 240.63 | 240.12 | 240.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 240.63 | 240.12 | 240.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:00:00 | 240.63 | 240.12 | 240.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 240.20 | 240.14 | 240.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 240.50 | 240.14 | 240.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 240.93 | 240.30 | 240.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:30:00 | 241.20 | 240.30 | 240.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 13:15:00 | 241.38 | 240.51 | 240.41 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 09:15:00 | 236.15 | 239.59 | 240.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 10:15:00 | 235.45 | 238.76 | 239.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 10:15:00 | 231.73 | 231.63 | 233.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-05 11:00:00 | 231.73 | 231.63 | 233.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 231.68 | 231.70 | 233.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:45:00 | 232.43 | 231.70 | 233.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 233.25 | 231.47 | 232.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 233.25 | 231.47 | 232.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 232.68 | 231.71 | 232.16 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 13:15:00 | 233.68 | 232.60 | 232.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 234.75 | 233.59 | 233.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 12:15:00 | 232.38 | 233.42 | 233.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 12:15:00 | 232.38 | 233.42 | 233.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 232.38 | 233.42 | 233.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 232.50 | 233.42 | 233.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 232.50 | 233.23 | 233.01 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 230.88 | 232.76 | 232.82 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 232.70 | 232.13 | 232.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 11:15:00 | 234.00 | 232.50 | 232.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 233.78 | 233.80 | 233.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 09:45:00 | 233.85 | 233.80 | 233.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 235.00 | 234.04 | 233.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:30:00 | 233.75 | 234.04 | 233.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 232.75 | 233.78 | 233.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:30:00 | 232.55 | 233.78 | 233.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 232.73 | 233.57 | 233.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:30:00 | 232.75 | 233.57 | 233.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 233.65 | 233.58 | 233.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:30:00 | 233.30 | 233.58 | 233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 233.15 | 233.49 | 233.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 234.25 | 233.49 | 233.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 10:15:00 | 232.80 | 233.32 | 233.19 | SL hit (close<static) qty=1.00 sl=233.13 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 11:15:00 | 232.10 | 233.07 | 233.09 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 14:15:00 | 233.58 | 233.09 | 233.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 15:15:00 | 234.00 | 233.27 | 233.17 | Break + close above crossover candle high |

### Cycle 10 — SELL (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 10:15:00 | 231.73 | 233.08 | 233.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 12:15:00 | 230.10 | 232.26 | 232.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 233.05 | 230.98 | 231.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 233.05 | 230.98 | 231.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 233.05 | 230.98 | 231.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 233.05 | 230.98 | 231.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 233.10 | 231.40 | 231.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 234.15 | 231.40 | 231.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 237.70 | 232.66 | 232.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 12:15:00 | 239.88 | 235.47 | 233.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 239.80 | 242.63 | 240.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 239.80 | 242.63 | 240.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 239.80 | 242.63 | 240.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 239.80 | 242.63 | 240.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 239.80 | 242.06 | 240.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:30:00 | 239.80 | 242.06 | 240.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 240.05 | 241.66 | 240.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 09:30:00 | 240.93 | 239.90 | 239.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 10:15:00 | 239.48 | 239.82 | 239.64 | SL hit (close<static) qty=1.00 sl=239.50 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 13:15:00 | 238.23 | 239.72 | 239.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 14:15:00 | 237.50 | 239.27 | 239.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 237.75 | 237.30 | 238.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-28 15:00:00 | 237.75 | 237.30 | 238.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 237.00 | 237.24 | 238.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:30:00 | 239.00 | 237.29 | 238.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 236.83 | 237.20 | 237.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 14:15:00 | 236.50 | 237.01 | 237.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 15:15:00 | 236.50 | 237.00 | 237.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 10:15:00 | 240.98 | 237.88 | 237.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 240.98 | 237.88 | 237.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 12:15:00 | 242.33 | 241.05 | 240.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 09:15:00 | 246.25 | 246.98 | 245.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 246.25 | 246.98 | 245.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 246.25 | 246.98 | 245.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 246.33 | 246.98 | 245.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 244.50 | 246.48 | 245.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 244.50 | 246.48 | 245.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 242.50 | 245.69 | 245.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 242.50 | 245.69 | 245.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 241.78 | 244.18 | 244.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 15:15:00 | 241.10 | 243.56 | 244.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 10:15:00 | 244.35 | 243.59 | 244.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 10:15:00 | 244.35 | 243.59 | 244.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 244.35 | 243.59 | 244.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 244.35 | 243.59 | 244.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 245.25 | 243.92 | 244.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:45:00 | 245.23 | 243.92 | 244.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 243.98 | 243.93 | 244.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 13:15:00 | 243.40 | 243.93 | 244.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 09:15:00 | 246.85 | 244.29 | 244.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 246.85 | 244.29 | 244.22 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 243.03 | 244.85 | 244.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 241.00 | 244.08 | 244.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 10:15:00 | 243.93 | 243.07 | 243.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 10:15:00 | 243.93 | 243.07 | 243.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 243.93 | 243.07 | 243.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:00:00 | 243.93 | 243.07 | 243.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 243.95 | 243.25 | 243.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 244.00 | 243.25 | 243.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 244.80 | 243.56 | 243.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:00:00 | 244.80 | 243.56 | 243.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 244.63 | 243.77 | 243.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:15:00 | 244.48 | 243.77 | 243.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 243.65 | 243.75 | 243.96 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 245.95 | 244.38 | 244.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 12:15:00 | 247.08 | 245.39 | 244.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 247.53 | 247.85 | 246.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 10:00:00 | 247.53 | 247.85 | 246.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 245.98 | 247.41 | 246.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:45:00 | 245.98 | 247.41 | 246.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 245.80 | 247.09 | 246.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:00:00 | 245.80 | 247.09 | 246.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 245.75 | 246.44 | 246.48 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 246.98 | 246.55 | 246.53 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 245.88 | 246.42 | 246.47 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 13:15:00 | 247.30 | 246.65 | 246.56 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 09:15:00 | 236.48 | 244.78 | 245.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 09:15:00 | 232.58 | 235.08 | 237.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 10:15:00 | 232.60 | 231.68 | 233.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 11:00:00 | 232.60 | 231.68 | 233.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 231.50 | 230.45 | 231.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 15:00:00 | 231.50 | 230.45 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 231.38 | 230.64 | 231.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 232.20 | 230.64 | 231.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 231.73 | 230.86 | 231.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:15:00 | 231.40 | 230.86 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 230.50 | 230.78 | 231.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:15:00 | 230.10 | 230.70 | 231.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 14:45:00 | 230.10 | 230.35 | 231.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-02 09:15:00 | 229.00 | 230.32 | 230.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 11:15:00 | 227.58 | 227.42 | 227.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 11:15:00 | 227.58 | 227.42 | 227.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 228.73 | 227.68 | 227.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 228.28 | 228.31 | 227.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 228.28 | 228.31 | 227.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 227.63 | 228.18 | 227.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 227.68 | 228.18 | 227.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 228.30 | 228.20 | 227.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 13:15:00 | 228.45 | 228.20 | 227.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 14:00:00 | 228.58 | 228.28 | 228.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 11:15:00 | 228.48 | 228.46 | 228.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 10:15:00 | 226.33 | 228.45 | 228.42 | SL hit (close<static) qty=1.00 sl=227.65 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 11:15:00 | 223.75 | 227.51 | 228.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 223.40 | 226.15 | 227.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 14:15:00 | 218.33 | 217.73 | 219.53 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 12:45:00 | 216.90 | 217.53 | 218.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 15:00:00 | 216.50 | 217.36 | 218.46 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:15:00 | 216.33 | 217.33 | 218.35 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 215.88 | 217.04 | 218.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:30:00 | 215.20 | 216.73 | 217.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 09:15:00 | 216.93 | 215.61 | 216.69 | SL hit (close>ema400) qty=1.00 sl=216.69 alert=retest1 |

### Cycle 25 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 219.13 | 217.37 | 217.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 15:15:00 | 219.63 | 218.13 | 217.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 222.33 | 222.34 | 221.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 13:30:00 | 222.30 | 222.34 | 221.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 222.00 | 222.27 | 221.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:30:00 | 222.15 | 222.27 | 221.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 222.30 | 222.28 | 221.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 222.73 | 222.28 | 221.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 222.05 | 222.23 | 221.57 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 220.00 | 221.24 | 221.32 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 15:15:00 | 223.43 | 221.33 | 221.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 224.80 | 222.03 | 221.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 233.18 | 233.54 | 231.48 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 09:15:00 | 234.28 | 233.54 | 231.48 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 231.95 | 232.82 | 231.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:00:00 | 231.95 | 232.82 | 231.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 233.25 | 232.91 | 231.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 14:30:00 | 235.00 | 233.04 | 232.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 230.55 | 232.46 | 231.94 | SL hit (close<ema400) qty=1.00 sl=231.94 alert=retest1 |

### Cycle 28 — SELL (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 15:15:00 | 231.30 | 231.75 | 231.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 11:15:00 | 230.30 | 231.31 | 231.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 09:15:00 | 234.55 | 231.11 | 231.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 09:15:00 | 234.55 | 231.11 | 231.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 234.55 | 231.11 | 231.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 10:00:00 | 234.55 | 231.11 | 231.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 234.45 | 231.78 | 231.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 12:15:00 | 237.50 | 233.30 | 232.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 09:15:00 | 239.75 | 239.98 | 237.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 09:45:00 | 240.03 | 239.98 | 237.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 233.48 | 238.42 | 238.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 233.85 | 238.42 | 238.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 235.68 | 237.87 | 237.84 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 233.33 | 236.96 | 237.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 232.20 | 236.01 | 236.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 232.75 | 232.60 | 234.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:00:00 | 232.75 | 232.60 | 234.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 233.40 | 232.15 | 233.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:00:00 | 233.40 | 232.15 | 233.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 234.18 | 232.56 | 233.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:30:00 | 234.13 | 232.56 | 233.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 233.85 | 232.82 | 233.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:15:00 | 234.03 | 232.82 | 233.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 235.55 | 233.36 | 233.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:45:00 | 235.40 | 233.36 | 233.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 235.25 | 233.74 | 233.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 236.43 | 234.28 | 233.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 235.85 | 236.22 | 235.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 235.85 | 236.22 | 235.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 235.85 | 236.22 | 235.36 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 233.78 | 234.88 | 234.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 231.40 | 234.06 | 234.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 228.93 | 228.21 | 229.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 228.93 | 228.21 | 229.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 228.93 | 228.21 | 229.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 12:30:00 | 228.73 | 228.21 | 229.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 228.05 | 228.18 | 229.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 228.60 | 228.18 | 229.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 228.75 | 228.17 | 228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 228.50 | 228.17 | 228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 228.98 | 228.33 | 228.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 227.95 | 228.25 | 228.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 10:00:00 | 228.23 | 228.10 | 228.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 230.75 | 227.72 | 227.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 230.75 | 227.72 | 227.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 233.20 | 228.82 | 227.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 230.23 | 230.83 | 229.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 230.23 | 230.83 | 229.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 230.23 | 230.83 | 229.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 230.23 | 230.83 | 229.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 227.85 | 230.24 | 229.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 227.85 | 230.24 | 229.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 227.65 | 229.72 | 229.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:00:00 | 227.65 | 229.72 | 229.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 09:15:00 | 227.90 | 229.10 | 229.15 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 230.10 | 229.13 | 229.03 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 227.83 | 229.24 | 229.28 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 13:15:00 | 230.10 | 229.38 | 229.33 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 12:15:00 | 228.70 | 229.30 | 229.35 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 230.50 | 229.42 | 229.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 231.10 | 229.75 | 229.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 12:15:00 | 229.68 | 230.19 | 229.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 12:15:00 | 229.68 | 230.19 | 229.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 229.68 | 230.19 | 229.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:00:00 | 229.68 | 230.19 | 229.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 230.23 | 230.20 | 229.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 14:30:00 | 230.90 | 230.16 | 229.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 15:15:00 | 231.50 | 230.16 | 229.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 09:15:00 | 230.75 | 239.01 | 239.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 230.75 | 239.01 | 239.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 11:15:00 | 229.50 | 235.81 | 238.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 199.88 | 199.37 | 204.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 14:45:00 | 199.88 | 199.37 | 204.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 190.25 | 189.99 | 191.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:00:00 | 190.25 | 189.99 | 191.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 194.00 | 190.88 | 191.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:00:00 | 194.00 | 190.88 | 191.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 194.00 | 191.50 | 191.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 194.00 | 191.50 | 191.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 12:15:00 | 196.05 | 192.78 | 192.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 10:15:00 | 198.30 | 195.25 | 193.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 15:15:00 | 200.25 | 200.38 | 198.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 09:15:00 | 199.68 | 200.38 | 198.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 200.63 | 200.66 | 199.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:45:00 | 199.95 | 200.66 | 199.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 199.38 | 200.40 | 199.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:00:00 | 199.38 | 200.40 | 199.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 200.15 | 200.35 | 199.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 15:15:00 | 200.35 | 200.35 | 199.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 09:15:00 | 196.93 | 199.67 | 199.19 | SL hit (close<static) qty=1.00 sl=199.28 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 11:15:00 | 197.75 | 198.80 | 198.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 195.90 | 197.48 | 198.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 14:15:00 | 197.28 | 197.14 | 197.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 14:15:00 | 197.28 | 197.14 | 197.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 197.28 | 197.14 | 197.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:45:00 | 197.80 | 197.14 | 197.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 197.53 | 197.22 | 197.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 196.80 | 197.22 | 197.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:45:00 | 196.43 | 197.22 | 197.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 10:15:00 | 196.90 | 197.22 | 197.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 11:00:00 | 197.00 | 197.18 | 197.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 197.03 | 196.82 | 197.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 197.03 | 196.82 | 197.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 197.10 | 196.88 | 197.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 199.05 | 196.88 | 197.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 199.00 | 197.30 | 197.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 199.00 | 197.30 | 197.40 | SL hit (close>static) qty=1.00 sl=197.93 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 14:15:00 | 194.50 | 193.96 | 193.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 195.78 | 194.34 | 194.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 13:15:00 | 194.43 | 194.78 | 194.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 13:15:00 | 194.43 | 194.78 | 194.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 194.43 | 194.78 | 194.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:00:00 | 194.43 | 194.78 | 194.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 194.88 | 194.80 | 194.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 195.68 | 194.77 | 194.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 10:30:00 | 195.03 | 194.85 | 194.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 12:45:00 | 195.25 | 194.99 | 194.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 14:45:00 | 195.10 | 195.13 | 194.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 195.48 | 195.76 | 195.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:15:00 | 195.00 | 195.76 | 195.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 194.33 | 195.47 | 195.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 194.33 | 195.47 | 195.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 195.00 | 195.38 | 195.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:30:00 | 194.35 | 195.38 | 195.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-28 14:15:00 | 194.75 | 195.15 | 195.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 14:15:00 | 194.75 | 195.15 | 195.19 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 195.88 | 195.22 | 195.19 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 193.18 | 194.82 | 195.02 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 199.83 | 195.60 | 195.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 201.85 | 198.30 | 196.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 200.73 | 200.79 | 199.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:45:00 | 200.35 | 200.79 | 199.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 200.30 | 201.19 | 200.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 200.30 | 201.19 | 200.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 200.73 | 201.10 | 200.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 201.30 | 201.15 | 200.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 199.70 | 202.12 | 201.89 | SL hit (close<static) qty=1.00 sl=200.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 200.95 | 201.61 | 201.68 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 202.55 | 201.80 | 201.76 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 14:15:00 | 200.95 | 201.73 | 201.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 15:15:00 | 200.78 | 201.54 | 201.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 198.23 | 197.93 | 198.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 198.23 | 197.93 | 198.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 198.23 | 197.93 | 198.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:30:00 | 198.78 | 197.93 | 198.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 198.50 | 198.04 | 198.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:15:00 | 199.05 | 198.04 | 198.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 199.40 | 198.32 | 198.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:00:00 | 199.40 | 198.32 | 198.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 198.28 | 198.31 | 198.81 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 201.80 | 199.29 | 199.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 12:15:00 | 202.78 | 201.32 | 200.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 201.43 | 202.44 | 201.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 201.43 | 202.44 | 201.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 201.43 | 202.44 | 201.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 201.43 | 202.44 | 201.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 203.30 | 202.61 | 201.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:45:00 | 203.55 | 202.79 | 201.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:45:00 | 203.85 | 203.56 | 202.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 198.83 | 202.92 | 202.76 | SL hit (close<static) qty=1.00 sl=201.25 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 199.33 | 202.20 | 202.45 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 203.73 | 202.25 | 202.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 206.23 | 203.68 | 202.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 205.48 | 205.63 | 204.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 205.48 | 205.63 | 204.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 205.48 | 205.63 | 204.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 205.55 | 205.63 | 204.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 206.35 | 205.70 | 205.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 09:30:00 | 206.35 | 205.70 | 205.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 207.63 | 206.90 | 206.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:45:00 | 209.38 | 207.73 | 206.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 211.48 | 208.31 | 207.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:15:00 | 209.40 | 210.09 | 209.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 15:15:00 | 212.28 | 213.18 | 213.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 212.28 | 213.18 | 213.21 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 214.15 | 213.37 | 213.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 10:15:00 | 214.45 | 213.59 | 213.40 | Break + close above crossover candle high |

### Cycle 56 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 207.60 | 213.13 | 213.43 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 213.75 | 211.45 | 211.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 11:15:00 | 214.50 | 212.06 | 211.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 10:15:00 | 215.38 | 215.49 | 214.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 10:30:00 | 216.08 | 215.49 | 214.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 214.20 | 215.22 | 214.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 214.20 | 215.22 | 214.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 216.30 | 215.44 | 214.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 14:45:00 | 216.78 | 215.72 | 214.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:45:00 | 216.75 | 216.11 | 215.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 12:15:00 | 217.40 | 216.11 | 215.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 12:45:00 | 216.95 | 216.23 | 215.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 216.40 | 216.48 | 215.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:15:00 | 215.63 | 216.48 | 215.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 213.35 | 215.86 | 215.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 213.35 | 215.86 | 215.55 | SL hit (close<static) qty=1.00 sl=213.70 alert=retest2 |

### Cycle 58 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 217.05 | 219.82 | 220.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 15:15:00 | 216.73 | 218.21 | 219.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 10:15:00 | 209.33 | 208.56 | 212.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 209.33 | 208.56 | 212.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 212.10 | 209.66 | 211.32 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 15:15:00 | 212.48 | 212.07 | 212.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 215.53 | 212.76 | 212.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 13:15:00 | 223.18 | 223.56 | 221.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 14:00:00 | 223.18 | 223.56 | 221.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 219.98 | 222.84 | 221.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 219.98 | 222.84 | 221.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 220.50 | 222.37 | 221.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 222.15 | 222.37 | 221.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 221.50 | 223.69 | 223.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 221.50 | 223.69 | 223.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 220.03 | 222.96 | 223.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 220.03 | 219.18 | 220.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-09 15:00:00 | 220.03 | 219.18 | 220.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 217.05 | 218.92 | 220.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:15:00 | 216.60 | 218.92 | 220.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:15:00 | 215.83 | 216.14 | 217.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:30:00 | 216.60 | 216.43 | 217.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 214.28 | 216.66 | 217.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 216.90 | 216.20 | 216.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:45:00 | 216.75 | 216.20 | 216.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 218.50 | 216.66 | 216.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 218.50 | 216.66 | 216.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 218.03 | 216.93 | 217.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 219.83 | 216.93 | 217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 221.78 | 217.90 | 217.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 221.78 | 217.90 | 217.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 10:15:00 | 223.30 | 218.98 | 218.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 221.38 | 221.50 | 219.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 221.38 | 221.50 | 219.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 221.38 | 221.50 | 219.99 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 09:15:00 | 218.13 | 220.02 | 220.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 10:15:00 | 217.65 | 219.55 | 219.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 13:15:00 | 219.73 | 219.40 | 219.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 13:15:00 | 219.73 | 219.40 | 219.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 219.73 | 219.40 | 219.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 13:45:00 | 219.60 | 219.40 | 219.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 219.68 | 219.46 | 219.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:15:00 | 219.60 | 219.46 | 219.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 219.60 | 219.49 | 219.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:15:00 | 219.18 | 219.49 | 219.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 219.83 | 219.55 | 219.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 218.93 | 219.55 | 219.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 14:00:00 | 219.00 | 219.52 | 219.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 15:15:00 | 218.38 | 219.52 | 219.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 09:45:00 | 218.08 | 219.31 | 219.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 219.95 | 219.44 | 219.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 220.45 | 219.64 | 219.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 11:15:00 | 220.45 | 219.64 | 219.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 15:15:00 | 221.08 | 220.20 | 219.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 11:15:00 | 220.48 | 220.64 | 220.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 12:00:00 | 220.48 | 220.64 | 220.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 221.38 | 220.79 | 220.34 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 15:15:00 | 218.58 | 220.05 | 220.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 217.53 | 219.55 | 219.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 210.25 | 209.20 | 211.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 13:00:00 | 210.25 | 209.20 | 211.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 214.10 | 210.32 | 211.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 214.10 | 210.32 | 211.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 212.50 | 210.75 | 211.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 213.48 | 210.75 | 211.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 213.28 | 211.26 | 211.68 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 213.70 | 212.14 | 212.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 12:15:00 | 215.45 | 212.80 | 212.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 214.83 | 224.55 | 222.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 214.83 | 224.55 | 222.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 214.83 | 224.55 | 222.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:45:00 | 215.85 | 224.55 | 222.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 214.73 | 222.58 | 221.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 214.73 | 222.58 | 221.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 214.30 | 220.93 | 221.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 212.90 | 214.91 | 216.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 214.48 | 213.12 | 214.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 214.48 | 213.12 | 214.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 214.48 | 213.12 | 214.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 210.10 | 213.33 | 214.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 199.59 | 202.06 | 203.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-20 13:15:00 | 202.25 | 201.60 | 202.70 | SL hit (close>ema200) qty=0.50 sl=201.60 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 205.03 | 202.94 | 202.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 205.83 | 203.52 | 203.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 220.78 | 221.87 | 220.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 15:00:00 | 220.78 | 221.87 | 220.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 220.48 | 221.60 | 220.41 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 12:15:00 | 218.68 | 219.81 | 219.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 217.50 | 219.35 | 219.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 221.35 | 219.57 | 219.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 221.35 | 219.57 | 219.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 221.35 | 219.57 | 219.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:45:00 | 221.43 | 219.57 | 219.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 225.45 | 220.74 | 220.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 11:15:00 | 233.30 | 223.25 | 221.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 234.65 | 236.29 | 232.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 14:00:00 | 234.65 | 236.29 | 232.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 234.40 | 237.55 | 236.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 235.35 | 237.55 | 236.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 234.40 | 236.92 | 235.95 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 232.05 | 235.10 | 235.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 231.90 | 234.46 | 234.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 09:15:00 | 239.80 | 235.13 | 235.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 239.80 | 235.13 | 235.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 239.80 | 235.13 | 235.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:00:00 | 239.80 | 235.13 | 235.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 10:15:00 | 238.08 | 235.72 | 235.43 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 10:15:00 | 234.00 | 235.36 | 235.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 11:15:00 | 232.13 | 234.71 | 235.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 13:15:00 | 218.73 | 218.60 | 221.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-22 13:30:00 | 219.05 | 218.60 | 221.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 225.23 | 219.97 | 221.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 225.23 | 219.97 | 221.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 223.80 | 220.74 | 221.67 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 13:15:00 | 224.95 | 222.50 | 222.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 227.05 | 223.90 | 223.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 226.00 | 226.04 | 224.72 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:15:00 | 227.10 | 226.04 | 224.72 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 11:45:00 | 226.90 | 226.26 | 225.17 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 223.75 | 225.76 | 225.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-25 12:15:00 | 223.75 | 225.76 | 225.04 | SL hit (close<ema400) qty=1.00 sl=225.04 alert=retest1 |

### Cycle 74 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 229.83 | 231.64 | 231.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 228.58 | 231.03 | 231.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 15:15:00 | 222.05 | 221.00 | 223.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:15:00 | 229.43 | 221.00 | 223.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 232.03 | 223.21 | 224.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 232.03 | 223.21 | 224.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 231.00 | 224.76 | 225.00 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 228.43 | 225.50 | 225.31 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 09:15:00 | 222.00 | 224.97 | 225.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 220.25 | 222.72 | 223.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 227.40 | 223.22 | 223.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 227.40 | 223.22 | 223.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 227.40 | 223.22 | 223.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 227.40 | 223.22 | 223.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 227.28 | 224.03 | 224.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:45:00 | 227.30 | 224.03 | 224.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 11:15:00 | 226.25 | 224.48 | 224.38 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 14:15:00 | 222.50 | 224.22 | 224.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 218.00 | 222.73 | 223.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 10:15:00 | 218.85 | 218.44 | 220.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 11:00:00 | 218.85 | 218.44 | 220.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 220.83 | 219.02 | 220.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:00:00 | 220.83 | 219.02 | 220.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 220.85 | 219.39 | 220.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:30:00 | 221.18 | 219.39 | 220.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 222.50 | 220.01 | 220.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 222.50 | 220.01 | 220.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 222.43 | 220.49 | 220.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 10:00:00 | 220.05 | 220.41 | 220.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 09:45:00 | 219.55 | 219.51 | 219.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:45:00 | 220.15 | 219.55 | 219.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 12:00:00 | 220.08 | 219.66 | 219.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 219.58 | 219.64 | 219.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:15:00 | 219.00 | 219.64 | 219.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:30:00 | 219.15 | 219.50 | 219.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 09:30:00 | 219.15 | 219.45 | 219.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 13:30:00 | 219.15 | 219.36 | 219.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 220.25 | 219.54 | 219.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:45:00 | 220.78 | 219.54 | 219.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 221.50 | 219.93 | 219.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 221.50 | 219.93 | 219.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 221.75 | 220.29 | 219.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 220.85 | 220.85 | 220.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 220.85 | 220.85 | 220.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 220.85 | 220.85 | 220.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:15:00 | 221.00 | 220.85 | 220.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 220.90 | 220.86 | 220.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 220.90 | 220.86 | 220.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 222.00 | 221.09 | 220.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 222.00 | 221.09 | 220.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 220.68 | 221.02 | 220.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 220.90 | 221.02 | 220.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 220.40 | 220.90 | 220.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 220.40 | 220.90 | 220.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 220.55 | 220.83 | 220.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 220.08 | 220.83 | 220.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 218.93 | 220.45 | 220.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 218.20 | 220.45 | 220.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 221.20 | 220.60 | 220.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 223.35 | 220.65 | 220.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 226.60 | 230.45 | 230.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 226.60 | 230.45 | 230.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 224.83 | 227.99 | 229.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 234.53 | 227.12 | 227.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 234.53 | 227.12 | 227.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 234.53 | 227.12 | 227.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 234.53 | 227.12 | 227.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 233.63 | 228.42 | 228.33 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 215.80 | 227.11 | 228.50 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 231.88 | 226.06 | 225.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 234.00 | 230.05 | 228.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 234.50 | 234.60 | 233.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 234.50 | 234.60 | 233.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 234.50 | 234.60 | 233.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 232.75 | 234.60 | 233.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 233.05 | 234.29 | 233.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 233.05 | 234.29 | 233.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 232.15 | 233.86 | 232.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 232.15 | 233.86 | 232.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 236.38 | 234.37 | 233.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:30:00 | 237.18 | 234.96 | 233.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 237.60 | 235.24 | 233.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 236.95 | 240.50 | 240.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 236.95 | 240.50 | 240.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 235.20 | 237.49 | 238.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 11:15:00 | 237.83 | 237.17 | 238.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 12:00:00 | 237.83 | 237.17 | 238.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 237.50 | 237.23 | 238.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:45:00 | 236.70 | 237.37 | 238.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 238.70 | 237.63 | 238.21 | SL hit (close>static) qty=1.00 sl=238.48 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 239.78 | 237.83 | 237.71 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 235.30 | 237.30 | 237.57 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 239.05 | 237.86 | 237.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 241.33 | 238.69 | 238.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 259.58 | 260.15 | 255.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 259.58 | 260.15 | 255.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 257.70 | 258.92 | 257.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 258.15 | 258.92 | 257.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 258.02 | 258.74 | 257.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 259.20 | 258.74 | 257.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 259.23 | 259.03 | 257.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 259.58 | 259.01 | 257.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 15:15:00 | 259.55 | 259.23 | 258.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 259.20 | 259.28 | 258.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 258.00 | 259.28 | 258.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 259.10 | 259.24 | 258.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:30:00 | 260.30 | 259.58 | 258.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 13:00:00 | 260.45 | 259.58 | 258.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:45:00 | 262.05 | 259.90 | 259.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:30:00 | 261.75 | 260.78 | 260.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 261.30 | 261.20 | 260.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 261.30 | 261.20 | 260.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 261.38 | 261.23 | 260.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:30:00 | 261.50 | 261.04 | 260.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 268.02 | 260.80 | 260.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 10:00:00 | 261.90 | 262.93 | 262.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 263.50 | 267.25 | 267.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 263.50 | 267.25 | 267.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 263.08 | 266.41 | 267.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 266.02 | 265.89 | 266.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 13:15:00 | 266.02 | 265.89 | 266.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 266.02 | 265.89 | 266.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 266.02 | 265.89 | 266.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 270.38 | 266.50 | 266.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 270.38 | 266.50 | 266.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 270.27 | 267.25 | 267.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 272.13 | 268.23 | 267.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 270.90 | 270.97 | 269.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 270.90 | 270.97 | 269.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 270.90 | 270.97 | 269.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 271.18 | 270.97 | 269.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 265.38 | 269.85 | 269.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 265.95 | 269.85 | 269.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 266.25 | 269.13 | 268.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 265.77 | 269.13 | 268.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 266.90 | 268.69 | 268.76 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 270.00 | 268.95 | 268.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 272.88 | 269.73 | 269.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 268.00 | 271.31 | 270.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 268.00 | 271.31 | 270.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 268.00 | 271.31 | 270.51 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 268.23 | 270.00 | 270.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 267.40 | 269.48 | 269.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 11:15:00 | 269.15 | 268.67 | 269.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 11:15:00 | 269.15 | 268.67 | 269.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 269.15 | 268.67 | 269.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 269.15 | 268.67 | 269.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 270.15 | 268.96 | 269.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:45:00 | 270.38 | 268.96 | 269.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 267.98 | 268.77 | 269.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 269.75 | 268.77 | 269.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 270.18 | 269.05 | 269.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 270.18 | 269.05 | 269.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 271.00 | 269.44 | 269.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 275.70 | 269.44 | 269.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 271.98 | 269.95 | 269.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 277.77 | 275.30 | 273.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 276.13 | 276.19 | 274.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 12:00:00 | 276.13 | 276.19 | 274.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 274.43 | 275.69 | 274.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 274.43 | 275.69 | 274.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 273.83 | 275.31 | 274.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 273.83 | 275.31 | 274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 274.25 | 275.10 | 274.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 274.88 | 274.96 | 274.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 272.77 | 274.31 | 274.22 | SL hit (close<static) qty=1.00 sl=273.58 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 273.18 | 274.08 | 274.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 272.20 | 273.71 | 273.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 271.73 | 266.27 | 267.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 271.73 | 266.27 | 267.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 271.73 | 266.27 | 267.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 272.45 | 266.27 | 267.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 273.45 | 267.70 | 268.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 273.45 | 267.70 | 268.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 273.25 | 269.66 | 269.23 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 268.30 | 270.71 | 270.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 267.65 | 270.10 | 270.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 268.70 | 268.65 | 269.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 15:00:00 | 268.70 | 268.65 | 269.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 268.77 | 268.41 | 269.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 270.15 | 268.41 | 269.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 268.60 | 268.45 | 269.16 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 271.70 | 269.63 | 269.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 274.15 | 270.76 | 270.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 270.70 | 271.87 | 270.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 13:15:00 | 270.70 | 271.87 | 270.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 270.70 | 271.87 | 270.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 270.70 | 271.87 | 270.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 269.90 | 271.47 | 270.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 269.85 | 271.47 | 270.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 271.15 | 271.41 | 270.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 268.95 | 271.13 | 270.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 269.38 | 270.78 | 270.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 269.38 | 270.78 | 270.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 269.95 | 270.61 | 270.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 268.58 | 270.03 | 270.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 270.15 | 269.98 | 270.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 270.15 | 269.98 | 270.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 270.15 | 269.98 | 270.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 269.45 | 269.98 | 270.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 272.05 | 270.48 | 270.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 272.05 | 270.48 | 270.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 272.98 | 271.36 | 270.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 273.55 | 273.91 | 272.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 13:45:00 | 274.43 | 273.91 | 272.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 272.33 | 273.68 | 272.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 272.38 | 273.68 | 272.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 272.50 | 273.44 | 272.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 271.85 | 273.44 | 272.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 274.05 | 273.54 | 273.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 274.85 | 273.54 | 273.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:45:00 | 276.27 | 274.38 | 273.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 271.55 | 273.57 | 273.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 271.55 | 273.57 | 273.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 270.05 | 272.87 | 273.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 266.75 | 263.60 | 265.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 266.75 | 263.60 | 265.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 266.75 | 263.60 | 265.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 266.75 | 263.60 | 265.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 265.55 | 263.99 | 265.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:30:00 | 265.00 | 264.32 | 265.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 269.45 | 266.53 | 266.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 269.45 | 266.53 | 266.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 271.95 | 269.26 | 267.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 274.30 | 275.64 | 273.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 274.30 | 275.64 | 273.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 274.05 | 275.32 | 273.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 274.23 | 275.32 | 273.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 274.98 | 275.25 | 273.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 273.75 | 275.25 | 273.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 274.20 | 275.04 | 273.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 274.20 | 275.04 | 273.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 273.23 | 274.68 | 273.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 275.60 | 274.68 | 273.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 10:45:00 | 274.58 | 274.88 | 274.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:00:00 | 275.00 | 274.90 | 274.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 271.77 | 276.49 | 276.36 | SL hit (close<static) qty=1.00 sl=272.35 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 270.38 | 275.27 | 275.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 267.40 | 271.05 | 273.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 267.73 | 267.26 | 269.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 267.73 | 267.26 | 269.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 267.73 | 267.26 | 269.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 270.05 | 267.26 | 269.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 269.93 | 267.87 | 269.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 269.93 | 267.87 | 269.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 270.20 | 268.34 | 269.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:45:00 | 269.05 | 268.72 | 269.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 268.00 | 269.03 | 269.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 14:15:00 | 264.68 | 262.97 | 262.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 264.68 | 262.97 | 262.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 15:15:00 | 265.45 | 263.47 | 263.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 10:15:00 | 271.85 | 272.77 | 269.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 11:00:00 | 271.85 | 272.77 | 269.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 268.35 | 271.57 | 269.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 268.35 | 271.57 | 269.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 267.50 | 270.76 | 269.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 267.95 | 270.76 | 269.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 269.98 | 269.72 | 269.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 269.23 | 269.72 | 269.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 268.35 | 269.44 | 269.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 267.18 | 269.44 | 269.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 267.75 | 269.10 | 269.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 267.75 | 269.10 | 269.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 268.40 | 268.96 | 269.02 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 270.85 | 269.34 | 269.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 271.50 | 269.77 | 269.40 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 09:15:00 | 266.27 | 269.07 | 269.11 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 270.90 | 268.96 | 268.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 271.45 | 269.81 | 269.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 274.93 | 275.18 | 273.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 274.93 | 275.18 | 273.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 274.90 | 275.11 | 273.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:15:00 | 273.93 | 275.11 | 273.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 273.93 | 274.88 | 273.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 271.18 | 274.88 | 273.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 270.38 | 273.98 | 273.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 270.38 | 273.98 | 273.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 270.38 | 273.26 | 273.26 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 274.95 | 273.23 | 273.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 275.40 | 273.66 | 273.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 274.58 | 274.85 | 274.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 274.58 | 274.85 | 274.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 274.58 | 274.85 | 274.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 273.58 | 274.85 | 274.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 275.25 | 274.93 | 274.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 274.55 | 274.93 | 274.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 277.50 | 278.32 | 277.33 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 275.33 | 276.61 | 276.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 11:15:00 | 274.98 | 276.28 | 276.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 15:15:00 | 271.45 | 271.38 | 273.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:15:00 | 272.00 | 271.38 | 273.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 273.50 | 271.81 | 273.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 273.50 | 271.81 | 273.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 273.05 | 272.05 | 273.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 11:30:00 | 271.83 | 271.64 | 272.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 12:15:00 | 258.24 | 263.41 | 266.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 262.50 | 261.39 | 264.18 | SL hit (close>ema200) qty=0.50 sl=261.39 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 212.18 | 209.22 | 208.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 213.60 | 210.68 | 209.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 210.43 | 210.98 | 210.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 14:15:00 | 210.43 | 210.98 | 210.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 210.43 | 210.98 | 210.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 210.43 | 210.98 | 210.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 210.00 | 210.79 | 210.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 209.45 | 210.56 | 210.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 209.08 | 210.26 | 209.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 209.23 | 210.26 | 209.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 209.60 | 209.70 | 209.70 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 210.18 | 209.79 | 209.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 211.13 | 210.13 | 209.91 | Break + close above crossover candle high |

### Cycle 114 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 207.38 | 209.67 | 209.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 206.80 | 209.09 | 209.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 207.58 | 206.66 | 207.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 207.58 | 206.66 | 207.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 207.58 | 206.66 | 207.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 207.58 | 206.66 | 207.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 210.35 | 207.40 | 207.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 210.35 | 207.40 | 207.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 211.00 | 208.12 | 208.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 212.88 | 209.07 | 208.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 13:15:00 | 218.55 | 219.09 | 216.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:45:00 | 218.05 | 219.09 | 216.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 221.03 | 219.47 | 217.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 15:15:00 | 221.75 | 219.47 | 217.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:30:00 | 221.58 | 220.17 | 217.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 221.53 | 220.17 | 217.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 214.10 | 218.97 | 218.46 | SL hit (close<static) qty=1.00 sl=216.78 alert=retest2 |

### Cycle 116 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 213.03 | 217.79 | 217.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 209.80 | 214.05 | 215.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 10:15:00 | 157.60 | 157.39 | 163.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:00:00 | 157.60 | 157.39 | 163.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 161.20 | 157.57 | 161.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:45:00 | 160.53 | 157.57 | 161.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 162.30 | 158.51 | 161.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:00:00 | 162.30 | 158.51 | 161.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 162.03 | 159.22 | 161.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:30:00 | 161.57 | 159.22 | 161.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 162.45 | 160.21 | 161.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 162.45 | 160.21 | 161.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 162.10 | 160.59 | 161.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:45:00 | 160.45 | 161.07 | 161.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 11:45:00 | 160.82 | 161.17 | 161.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 14:00:00 | 160.95 | 161.13 | 161.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 161.00 | 160.02 | 160.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 159.55 | 159.93 | 160.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 158.93 | 159.30 | 159.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 163.78 | 160.05 | 159.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 163.78 | 160.05 | 159.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 15:15:00 | 165.00 | 161.04 | 160.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 189.15 | 191.12 | 187.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 10:00:00 | 189.15 | 191.12 | 187.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 192.58 | 193.36 | 192.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 192.58 | 193.36 | 192.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 191.73 | 193.04 | 192.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:00:00 | 191.73 | 193.04 | 192.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 192.85 | 193.00 | 192.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 194.58 | 193.09 | 192.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:45:00 | 193.90 | 193.83 | 193.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 194.35 | 193.75 | 193.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 194.63 | 193.32 | 193.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 195.45 | 193.75 | 193.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 196.35 | 194.74 | 193.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:45:00 | 196.98 | 195.64 | 194.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 192.80 | 195.72 | 195.58 | SL hit (close<static) qty=1.00 sl=193.28 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 192.05 | 194.98 | 195.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 191.23 | 194.23 | 194.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 13:15:00 | 194.80 | 192.24 | 193.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 13:15:00 | 194.80 | 192.24 | 193.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 194.80 | 192.24 | 193.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 194.80 | 192.24 | 193.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 198.10 | 193.41 | 193.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 197.65 | 193.41 | 193.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 15:15:00 | 200.95 | 194.92 | 194.35 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 193.35 | 194.46 | 194.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 192.50 | 194.07 | 194.38 | Break + close below crossover candle low |

### Cycle 121 — BUY (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 09:15:00 | 197.65 | 194.78 | 194.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 09:15:00 | 201.58 | 196.24 | 195.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 198.13 | 198.46 | 197.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 09:15:00 | 198.38 | 198.46 | 197.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 200.35 | 198.84 | 197.44 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 195.30 | 196.85 | 196.98 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 199.13 | 196.84 | 196.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 13:15:00 | 200.30 | 197.76 | 197.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 204.33 | 205.21 | 202.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 10:00:00 | 204.33 | 205.21 | 202.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 214.25 | 217.94 | 214.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 214.25 | 217.94 | 214.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 212.48 | 216.85 | 213.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:45:00 | 213.75 | 216.85 | 213.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 212.75 | 216.03 | 213.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 212.00 | 216.03 | 213.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 213.18 | 215.46 | 213.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:30:00 | 214.73 | 213.28 | 213.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 210.53 | 212.73 | 212.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 10:15:00 | 210.53 | 212.73 | 212.86 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 214.93 | 213.17 | 213.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 215.73 | 214.18 | 213.58 | Break + close above crossover candle high |

### Cycle 126 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 207.85 | 213.08 | 213.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 205.15 | 210.51 | 211.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 215.98 | 208.64 | 210.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 215.98 | 208.64 | 210.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 215.98 | 208.64 | 210.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 215.98 | 208.64 | 210.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 214.33 | 209.78 | 210.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:15:00 | 212.60 | 209.78 | 210.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 212.78 | 210.55 | 210.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 214.00 | 211.24 | 211.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 214.00 | 211.24 | 211.06 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 208.80 | 211.09 | 211.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 202.95 | 206.74 | 208.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 206.85 | 204.48 | 206.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 206.85 | 204.48 | 206.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 206.85 | 204.48 | 206.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 203.45 | 205.09 | 206.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:45:00 | 203.03 | 204.83 | 205.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:00:00 | 203.40 | 204.54 | 205.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:30:00 | 203.50 | 204.11 | 205.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 203.83 | 202.93 | 204.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:00:00 | 202.08 | 202.76 | 203.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 14:00:00 | 202.35 | 202.07 | 202.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:00:00 | 202.38 | 202.13 | 202.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 198.28 | 202.31 | 202.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 199.50 | 201.75 | 202.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 197.55 | 199.58 | 200.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 193.28 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 192.88 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 193.23 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 193.32 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 192.23 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 192.26 | 197.84 | 199.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 191.98 | 196.45 | 198.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 194.18 | 193.73 | 196.54 | SL hit (close>ema200) qty=0.50 sl=193.73 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 198.70 | 192.36 | 191.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 203.95 | 201.31 | 199.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 199.67 | 201.02 | 199.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 199.67 | 201.02 | 199.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 199.67 | 201.02 | 199.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 200.07 | 201.02 | 199.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 195.43 | 199.91 | 198.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 195.43 | 199.91 | 198.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 196.85 | 199.29 | 198.79 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 194.11 | 198.26 | 198.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 192.92 | 196.60 | 197.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 194.12 | 191.67 | 193.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 14:15:00 | 194.12 | 191.67 | 193.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 194.12 | 191.67 | 193.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 194.12 | 191.67 | 193.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 194.00 | 192.14 | 193.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 198.85 | 192.14 | 193.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 204.72 | 194.65 | 194.43 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 198.86 | 200.25 | 200.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 196.67 | 199.43 | 199.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 190.80 | 190.71 | 192.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 190.57 | 190.71 | 192.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 193.29 | 191.23 | 192.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 193.74 | 191.23 | 192.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 195.18 | 192.02 | 192.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 195.18 | 192.02 | 192.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 188.26 | 191.41 | 192.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 187.81 | 191.41 | 192.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:45:00 | 187.30 | 187.28 | 189.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 12:15:00 | 187.96 | 187.54 | 189.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 187.37 | 188.93 | 189.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 188.22 | 188.91 | 189.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:30:00 | 187.58 | 188.65 | 189.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:15:00 | 187.53 | 188.65 | 189.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 192.63 | 189.28 | 189.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 192.63 | 189.28 | 189.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 193.49 | 190.56 | 189.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 200.64 | 201.52 | 199.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 199.58 | 201.52 | 199.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 195.46 | 200.31 | 198.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 195.46 | 200.31 | 198.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 199.05 | 200.06 | 198.97 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 197.80 | 198.46 | 198.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 194.91 | 197.06 | 197.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 195.92 | 195.41 | 196.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 195.92 | 195.41 | 196.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 195.57 | 195.44 | 196.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 191.79 | 195.44 | 196.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 11:15:00 | 182.20 | 187.64 | 190.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 187.97 | 187.71 | 190.62 | SL hit (close>ema200) qty=0.50 sl=187.71 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 189.54 | 184.73 | 184.72 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 184.31 | 186.50 | 186.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 183.19 | 185.84 | 186.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 185.40 | 185.04 | 185.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 185.40 | 185.04 | 185.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 185.40 | 185.04 | 185.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 185.40 | 185.04 | 185.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 186.14 | 185.01 | 185.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 186.14 | 185.01 | 185.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 187.13 | 185.43 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 187.02 | 185.43 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 187.40 | 185.82 | 185.82 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 185.13 | 185.69 | 185.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 184.45 | 185.44 | 185.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 187.21 | 185.64 | 185.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 187.21 | 185.64 | 185.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 187.21 | 185.64 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 187.21 | 185.64 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 188.60 | 186.23 | 185.93 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 185.01 | 185.98 | 186.09 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 188.15 | 186.41 | 186.27 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 184.83 | 185.95 | 186.09 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 188.04 | 186.37 | 186.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 188.87 | 187.44 | 186.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 196.95 | 197.25 | 195.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 196.95 | 197.25 | 195.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 198.38 | 197.48 | 195.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 194.50 | 197.48 | 195.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 199.15 | 201.58 | 199.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 199.15 | 201.58 | 199.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 201.55 | 201.58 | 199.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 198.13 | 201.58 | 199.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 199.25 | 201.11 | 199.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 199.25 | 201.11 | 199.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 200.76 | 201.04 | 199.99 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 197.41 | 199.23 | 199.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 196.23 | 198.63 | 199.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 196.95 | 196.93 | 197.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:00:00 | 196.95 | 196.93 | 197.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 197.77 | 197.10 | 197.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 197.35 | 197.10 | 197.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 196.79 | 197.04 | 197.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 13:15:00 | 196.71 | 197.04 | 197.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 13:45:00 | 196.12 | 196.74 | 197.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 202.08 | 197.42 | 197.68 | SL hit (close>static) qty=1.00 sl=198.34 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 201.71 | 198.28 | 198.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 15:15:00 | 203.78 | 201.41 | 199.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 200.94 | 201.71 | 200.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 200.94 | 201.71 | 200.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 200.94 | 201.71 | 200.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 200.94 | 201.71 | 200.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 201.98 | 201.76 | 200.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 200.57 | 201.76 | 200.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 200.70 | 201.55 | 200.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:00:00 | 200.70 | 201.55 | 200.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 202.12 | 201.67 | 200.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 204.38 | 201.47 | 201.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 198.28 | 202.50 | 202.48 | SL hit (close<static) qty=1.00 sl=200.30 alert=retest2 |

### Cycle 146 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 197.61 | 201.53 | 202.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 196.77 | 200.57 | 201.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 189.90 | 189.32 | 192.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 189.90 | 189.32 | 192.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 178.71 | 175.96 | 178.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 09:15:00 | 174.33 | 177.94 | 178.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 175.23 | 176.30 | 177.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 175.75 | 176.28 | 177.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 09:45:00 | 175.58 | 176.09 | 177.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 177.27 | 176.45 | 177.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:30:00 | 177.19 | 176.45 | 177.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 177.84 | 176.73 | 177.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 177.84 | 176.73 | 177.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 177.65 | 176.92 | 177.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:15:00 | 177.75 | 176.92 | 177.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 176.84 | 176.90 | 177.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:30:00 | 177.78 | 176.90 | 177.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 177.01 | 176.92 | 177.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 180.35 | 176.92 | 177.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 180.46 | 177.63 | 177.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 180.46 | 177.63 | 177.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 181.45 | 178.39 | 177.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 185.00 | 185.94 | 183.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 185.00 | 185.94 | 183.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 183.55 | 185.12 | 183.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 183.55 | 185.12 | 183.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 185.35 | 185.17 | 183.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 185.67 | 185.27 | 183.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 185.58 | 185.22 | 184.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:30:00 | 186.85 | 185.71 | 184.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 185.75 | 185.53 | 184.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 181.96 | 184.85 | 184.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 181.96 | 184.85 | 184.62 | SL hit (close<static) qty=1.00 sl=182.80 alert=retest2 |

### Cycle 148 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 179.52 | 183.79 | 184.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 178.41 | 181.18 | 182.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 183.71 | 181.24 | 182.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 183.71 | 181.24 | 182.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 183.71 | 181.24 | 182.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 182.49 | 181.24 | 182.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 182.99 | 181.59 | 182.47 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 185.50 | 183.30 | 183.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 188.50 | 184.85 | 183.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 185.74 | 185.84 | 184.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:00:00 | 185.74 | 185.84 | 184.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 185.16 | 185.71 | 184.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 185.16 | 185.71 | 184.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 185.06 | 185.58 | 184.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 186.83 | 185.58 | 184.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 189.72 | 186.41 | 185.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 10:45:00 | 191.09 | 187.31 | 185.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 198.80 | 202.58 | 202.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 198.80 | 202.58 | 202.86 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 204.01 | 202.11 | 201.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 205.51 | 203.37 | 202.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 203.43 | 203.80 | 203.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 203.43 | 203.80 | 203.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 203.43 | 203.80 | 203.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 203.43 | 203.80 | 203.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 202.46 | 203.53 | 203.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 202.46 | 203.53 | 203.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 203.34 | 203.49 | 203.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 204.19 | 203.45 | 203.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 203.90 | 204.70 | 204.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 202.90 | 203.98 | 204.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 12:15:00 | 202.90 | 203.98 | 204.04 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 204.65 | 204.15 | 204.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 207.33 | 204.95 | 204.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 211.20 | 211.29 | 209.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:15:00 | 212.99 | 211.29 | 209.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:45:00 | 212.30 | 211.57 | 209.70 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 209.35 | 211.17 | 209.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 209.35 | 211.17 | 209.84 | SL hit (close<ema400) qty=1.00 sl=209.84 alert=retest1 |

### Cycle 154 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 205.76 | 208.90 | 209.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 203.23 | 205.84 | 206.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 205.30 | 204.77 | 205.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 205.30 | 204.77 | 205.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 205.30 | 204.77 | 205.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 205.71 | 204.77 | 205.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 206.06 | 205.03 | 205.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 206.06 | 205.03 | 205.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 206.35 | 205.29 | 205.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 206.00 | 205.29 | 205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 205.16 | 205.26 | 205.87 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 206.48 | 206.03 | 205.99 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 203.84 | 205.62 | 205.81 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 210.02 | 206.67 | 206.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 212.82 | 207.90 | 206.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 209.27 | 212.84 | 211.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 209.27 | 212.84 | 211.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 209.27 | 212.84 | 211.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 209.40 | 212.84 | 211.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 208.02 | 211.87 | 211.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 208.13 | 211.87 | 211.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 208.57 | 210.31 | 210.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 206.42 | 209.54 | 210.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 213.19 | 209.84 | 210.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 213.19 | 209.84 | 210.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 213.19 | 209.84 | 210.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 216.00 | 209.84 | 210.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 210.60 | 209.99 | 210.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 209.40 | 209.99 | 210.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 15:15:00 | 208.93 | 208.12 | 208.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 208.93 | 208.12 | 208.09 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 207.45 | 207.98 | 208.03 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 208.80 | 208.10 | 208.07 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 207.41 | 207.96 | 208.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 207.15 | 207.80 | 207.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 14:15:00 | 208.37 | 207.91 | 207.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 14:15:00 | 208.37 | 207.91 | 207.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 208.37 | 207.91 | 207.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 208.37 | 207.91 | 207.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 208.37 | 208.00 | 208.01 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 208.73 | 208.15 | 208.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 209.50 | 208.62 | 208.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 213.40 | 213.95 | 212.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:30:00 | 213.81 | 213.95 | 212.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 213.36 | 213.83 | 212.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:15:00 | 213.00 | 213.83 | 212.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 212.69 | 213.60 | 212.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 212.69 | 213.60 | 212.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 211.00 | 213.08 | 212.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 211.04 | 213.08 | 212.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 211.60 | 212.41 | 212.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 207.37 | 211.40 | 212.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 202.83 | 200.37 | 203.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 202.83 | 200.37 | 203.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 202.83 | 200.37 | 203.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 202.83 | 200.37 | 203.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 207.25 | 201.75 | 203.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 207.25 | 201.75 | 203.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 205.90 | 202.58 | 203.87 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 210.86 | 205.01 | 204.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 211.62 | 206.33 | 205.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 208.83 | 208.92 | 207.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 13:45:00 | 208.75 | 208.92 | 207.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 207.43 | 208.70 | 207.72 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 205.14 | 206.93 | 207.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 204.31 | 206.04 | 206.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 205.26 | 203.86 | 204.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 205.26 | 203.86 | 204.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 205.26 | 203.86 | 204.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 205.21 | 203.86 | 204.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 205.97 | 204.29 | 205.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 205.97 | 204.29 | 205.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 205.62 | 204.55 | 205.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 205.07 | 204.75 | 205.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 209.28 | 205.95 | 205.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 209.28 | 205.95 | 205.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 212.74 | 208.99 | 208.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 215.83 | 215.94 | 213.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 215.83 | 215.94 | 213.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 221.17 | 220.20 | 218.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:30:00 | 223.00 | 220.79 | 219.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 223.57 | 224.60 | 224.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 223.57 | 224.60 | 224.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 222.52 | 224.18 | 224.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 218.70 | 217.92 | 219.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:15:00 | 219.18 | 217.92 | 219.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 219.54 | 218.24 | 219.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 219.54 | 218.24 | 219.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 219.67 | 218.53 | 219.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 219.40 | 218.53 | 219.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 219.82 | 218.78 | 219.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 219.82 | 218.78 | 219.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 220.20 | 219.13 | 219.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 220.20 | 219.13 | 219.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 220.80 | 219.47 | 219.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 221.70 | 219.91 | 219.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 219.89 | 219.97 | 219.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 219.89 | 219.97 | 219.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 219.89 | 219.97 | 219.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 219.89 | 219.97 | 219.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 219.35 | 219.84 | 219.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 219.88 | 219.84 | 219.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 220.60 | 219.99 | 219.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 220.91 | 219.94 | 219.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:30:00 | 220.84 | 219.99 | 219.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 219.00 | 219.79 | 219.78 | SL hit (close<static) qty=1.00 sl=219.33 alert=retest2 |

### Cycle 170 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 219.20 | 219.67 | 219.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 216.33 | 219.00 | 219.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 213.88 | 213.34 | 214.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 213.88 | 213.34 | 214.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 213.80 | 213.41 | 214.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 215.56 | 213.41 | 214.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 213.59 | 213.29 | 214.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 212.22 | 213.03 | 213.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:15:00 | 201.61 | 205.27 | 207.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 206.06 | 203.95 | 206.34 | SL hit (close>ema200) qty=0.50 sl=203.95 alert=retest2 |

### Cycle 171 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 206.33 | 204.75 | 204.67 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 203.30 | 204.53 | 204.60 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 206.11 | 204.26 | 204.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 206.90 | 205.11 | 204.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 10:15:00 | 204.86 | 205.08 | 204.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 204.86 | 205.08 | 204.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 204.86 | 205.08 | 204.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 204.40 | 205.08 | 204.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 205.13 | 205.82 | 205.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 205.13 | 205.82 | 205.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 204.79 | 205.61 | 205.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 204.79 | 205.61 | 205.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 205.11 | 205.51 | 205.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 205.25 | 205.51 | 205.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 203.81 | 205.02 | 205.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 203.81 | 205.02 | 205.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 202.59 | 204.54 | 204.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 205.65 | 203.56 | 204.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 205.65 | 203.56 | 204.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 205.65 | 203.56 | 204.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 205.65 | 203.56 | 204.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 205.89 | 204.02 | 204.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 205.69 | 204.02 | 204.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 203.89 | 204.04 | 204.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 202.90 | 203.78 | 204.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 202.40 | 203.26 | 203.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:45:00 | 202.87 | 202.41 | 202.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 207.25 | 203.34 | 203.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 207.25 | 203.34 | 203.28 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 204.28 | 205.23 | 205.23 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 206.23 | 205.17 | 205.16 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 204.58 | 205.05 | 205.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 13:15:00 | 204.35 | 204.93 | 205.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 15:15:00 | 205.00 | 204.92 | 205.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 15:15:00 | 205.00 | 204.92 | 205.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 205.00 | 204.92 | 205.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 204.26 | 204.92 | 205.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 204.54 | 204.84 | 204.98 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 207.36 | 205.30 | 205.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 207.94 | 205.83 | 205.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 205.93 | 206.47 | 205.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 205.93 | 206.47 | 205.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 205.93 | 206.47 | 205.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 205.93 | 206.47 | 205.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 206.35 | 206.45 | 205.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 207.29 | 206.45 | 205.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:45:00 | 207.99 | 208.76 | 208.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 208.33 | 208.67 | 208.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 208.33 | 208.67 | 208.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 207.36 | 208.28 | 208.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 207.91 | 207.36 | 207.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 207.91 | 207.36 | 207.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 207.91 | 207.36 | 207.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 207.91 | 207.36 | 207.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 207.66 | 207.42 | 207.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 206.98 | 207.31 | 207.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 206.98 | 207.22 | 207.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 209.90 | 207.72 | 207.76 | SL hit (close>static) qty=1.00 sl=208.40 alert=retest2 |

### Cycle 181 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 210.76 | 208.33 | 208.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 210.90 | 209.46 | 208.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 217.97 | 218.11 | 215.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 213.40 | 216.71 | 216.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 213.40 | 216.71 | 216.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 213.40 | 216.71 | 216.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 213.08 | 215.98 | 215.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 212.76 | 215.98 | 215.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 211.90 | 215.17 | 215.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 211.71 | 213.35 | 214.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 211.14 | 210.83 | 212.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:45:00 | 210.49 | 210.83 | 212.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 211.50 | 210.96 | 211.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 214.73 | 210.96 | 211.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 213.76 | 211.52 | 211.84 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 214.64 | 212.14 | 212.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 215.97 | 214.18 | 213.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 214.18 | 214.25 | 213.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 214.18 | 214.25 | 213.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 214.19 | 214.40 | 213.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 215.59 | 214.57 | 214.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 215.81 | 214.74 | 214.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 215.55 | 215.02 | 214.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 215.55 | 215.16 | 214.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 215.21 | 215.45 | 215.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 215.05 | 215.45 | 215.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 215.15 | 215.39 | 215.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 215.56 | 215.24 | 215.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 215.64 | 215.25 | 215.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 214.74 | 215.24 | 215.12 | SL hit (close<static) qty=1.00 sl=214.86 alert=retest2 |

### Cycle 184 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 214.55 | 215.21 | 215.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 213.77 | 214.79 | 215.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 205.82 | 203.65 | 205.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 205.82 | 203.65 | 205.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 205.82 | 203.65 | 205.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 205.82 | 203.65 | 205.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 207.24 | 204.37 | 205.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 207.24 | 204.37 | 205.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 206.70 | 206.02 | 205.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 208.70 | 207.13 | 206.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 208.37 | 208.38 | 207.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 11:45:00 | 208.44 | 208.38 | 207.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 205.70 | 208.03 | 207.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 205.70 | 208.03 | 207.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 205.57 | 207.54 | 207.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 205.63 | 207.54 | 207.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 11:15:00 | 205.37 | 207.11 | 207.29 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 209.51 | 207.58 | 207.42 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 207.22 | 207.38 | 207.38 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 207.96 | 207.50 | 207.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 210.21 | 208.11 | 207.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 216.50 | 216.54 | 213.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 216.50 | 216.54 | 213.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 217.14 | 218.03 | 216.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 215.73 | 218.03 | 216.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 216.25 | 217.67 | 216.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 215.69 | 217.67 | 216.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 216.72 | 217.48 | 216.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 216.50 | 217.48 | 216.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 217.00 | 217.39 | 216.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:30:00 | 216.90 | 217.39 | 216.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 216.39 | 217.19 | 216.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 216.39 | 217.19 | 216.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 216.20 | 216.99 | 216.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 216.26 | 216.99 | 216.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 216.05 | 216.80 | 216.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 214.64 | 216.80 | 216.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 215.72 | 216.26 | 216.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 214.49 | 215.85 | 216.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 11:15:00 | 212.82 | 212.61 | 213.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:45:00 | 212.70 | 212.61 | 213.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 212.24 | 212.24 | 213.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 211.12 | 212.95 | 213.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 210.60 | 212.48 | 212.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 210.58 | 211.98 | 212.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 213.70 | 210.25 | 209.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 213.70 | 210.25 | 209.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 214.90 | 211.18 | 210.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 212.09 | 212.35 | 211.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 212.09 | 212.35 | 211.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 211.29 | 212.14 | 211.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:15:00 | 210.84 | 212.14 | 211.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 210.66 | 211.84 | 211.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 210.86 | 211.84 | 211.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 210.84 | 211.64 | 211.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 210.82 | 211.64 | 211.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 211.45 | 211.49 | 211.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 211.37 | 211.49 | 211.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 211.78 | 211.55 | 211.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:30:00 | 213.20 | 211.98 | 211.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 212.50 | 212.68 | 212.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 209.39 | 211.54 | 211.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 209.39 | 211.54 | 211.78 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 214.46 | 212.12 | 211.94 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 210.78 | 211.90 | 212.03 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 213.07 | 211.94 | 211.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 214.49 | 212.55 | 212.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 13:15:00 | 212.66 | 213.06 | 212.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 212.66 | 213.06 | 212.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 212.66 | 213.06 | 212.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 212.66 | 213.06 | 212.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 214.84 | 213.42 | 212.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 218.06 | 213.83 | 213.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 211.19 | 214.10 | 213.84 | SL hit (close<static) qty=1.00 sl=212.38 alert=retest2 |

### Cycle 196 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 210.86 | 213.45 | 213.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 210.70 | 211.95 | 212.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 211.63 | 211.44 | 212.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:30:00 | 211.23 | 211.44 | 212.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 212.44 | 211.64 | 212.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 212.44 | 211.64 | 212.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 212.29 | 211.77 | 212.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 212.33 | 211.77 | 212.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 212.00 | 211.81 | 212.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:30:00 | 211.38 | 211.79 | 212.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 211.29 | 211.69 | 212.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 209.39 | 209.77 | 210.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 215.05 | 210.87 | 210.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 215.05 | 210.87 | 210.51 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 211.00 | 212.36 | 212.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 209.45 | 211.54 | 212.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 206.19 | 203.21 | 204.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 206.19 | 203.21 | 204.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 206.19 | 203.21 | 204.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 206.19 | 203.21 | 204.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 205.74 | 203.71 | 204.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:30:00 | 204.75 | 203.93 | 204.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 194.51 | 198.69 | 201.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 197.15 | 196.85 | 199.39 | SL hit (close>ema200) qty=0.50 sl=196.85 alert=retest2 |

### Cycle 199 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 203.15 | 199.26 | 199.11 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 198.50 | 199.59 | 199.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 198.20 | 198.97 | 199.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 194.58 | 194.36 | 196.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 194.58 | 194.36 | 196.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 186.70 | 185.33 | 187.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 187.29 | 185.33 | 187.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 185.82 | 185.43 | 187.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 185.30 | 185.43 | 187.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 185.15 | 185.42 | 186.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 186.50 | 185.79 | 185.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 186.50 | 185.79 | 185.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 187.35 | 186.32 | 186.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 185.72 | 186.20 | 185.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 185.72 | 186.20 | 185.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 185.72 | 186.20 | 185.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 185.72 | 186.20 | 185.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 185.78 | 186.11 | 185.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 185.95 | 186.11 | 185.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 185.51 | 185.87 | 185.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 185.51 | 185.87 | 185.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 184.69 | 185.55 | 185.73 | Break + close below crossover candle low |

### Cycle 203 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 191.10 | 185.48 | 185.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 11:15:00 | 195.55 | 188.57 | 186.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 192.54 | 194.58 | 192.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 192.54 | 194.58 | 192.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 192.54 | 194.58 | 192.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 196.25 | 193.88 | 192.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 193.50 | 194.81 | 194.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 193.50 | 194.81 | 194.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 193.37 | 194.27 | 194.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 194.00 | 193.71 | 194.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 194.00 | 193.71 | 194.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 194.00 | 193.71 | 194.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 193.87 | 193.71 | 194.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 194.02 | 193.77 | 194.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 194.08 | 193.77 | 194.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 194.00 | 193.82 | 194.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 194.00 | 193.82 | 194.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 194.00 | 193.86 | 194.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:45:00 | 194.03 | 193.86 | 194.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 195.07 | 194.10 | 194.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 195.07 | 194.10 | 194.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 194.63 | 194.21 | 194.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 197.40 | 194.84 | 194.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 190.50 | 194.71 | 194.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 190.50 | 194.71 | 194.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 190.50 | 194.71 | 194.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 190.50 | 194.71 | 194.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 191.10 | 193.99 | 194.35 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 194.50 | 193.80 | 193.75 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 191.11 | 193.26 | 193.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 191.00 | 192.81 | 193.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 186.96 | 185.58 | 186.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 14:15:00 | 186.96 | 185.58 | 186.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 186.96 | 185.58 | 186.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 186.96 | 185.58 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 187.35 | 185.94 | 186.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 184.61 | 185.94 | 186.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 183.31 | 185.38 | 186.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 182.20 | 185.38 | 186.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 12:45:00 | 183.24 | 184.60 | 185.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 183.06 | 184.60 | 185.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 183.20 | 184.03 | 185.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 182.14 | 181.94 | 183.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 181.20 | 181.83 | 182.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 179.32 | 178.60 | 178.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 179.32 | 178.60 | 178.52 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 177.21 | 178.40 | 178.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 176.32 | 177.99 | 178.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 175.34 | 174.65 | 175.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 175.34 | 174.65 | 175.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 175.34 | 174.65 | 175.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 176.64 | 174.65 | 175.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 176.04 | 174.93 | 175.77 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 179.05 | 175.94 | 175.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 179.92 | 177.77 | 176.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 178.00 | 178.50 | 177.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 178.00 | 178.50 | 177.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 178.00 | 178.50 | 177.48 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 174.81 | 177.01 | 177.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 174.31 | 175.82 | 176.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 173.00 | 172.99 | 174.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 173.00 | 172.99 | 174.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 174.15 | 173.23 | 174.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 172.70 | 173.23 | 174.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 174.24 | 173.05 | 173.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 174.24 | 173.05 | 173.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 174.80 | 173.74 | 173.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 176.76 | 177.09 | 176.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:30:00 | 176.70 | 177.09 | 176.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 175.20 | 176.77 | 176.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 175.20 | 176.77 | 176.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 175.81 | 176.58 | 176.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 176.24 | 176.28 | 176.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 176.49 | 176.19 | 176.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 169.32 | 174.79 | 175.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 169.32 | 174.79 | 175.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 163.61 | 169.63 | 172.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 168.27 | 168.26 | 170.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:45:00 | 167.25 | 168.26 | 170.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 170.20 | 168.18 | 169.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 169.94 | 168.18 | 169.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 169.81 | 168.50 | 169.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 171.25 | 168.50 | 169.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 171.03 | 169.01 | 170.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 171.03 | 169.01 | 170.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 172.23 | 169.65 | 170.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 172.17 | 169.65 | 170.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 171.98 | 170.63 | 170.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 172.50 | 171.54 | 171.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 169.89 | 171.32 | 171.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 169.89 | 171.32 | 171.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 169.89 | 171.32 | 171.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 170.02 | 171.32 | 171.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 169.00 | 170.86 | 170.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 15:15:00 | 168.71 | 169.55 | 170.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 167.96 | 167.67 | 168.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 167.96 | 167.67 | 168.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 168.25 | 167.79 | 168.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 168.25 | 167.79 | 168.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 168.59 | 168.01 | 168.46 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 169.80 | 168.68 | 168.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 170.37 | 169.21 | 168.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 168.95 | 169.87 | 169.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 168.95 | 169.87 | 169.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 168.95 | 169.87 | 169.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:30:00 | 168.87 | 169.87 | 169.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 168.85 | 169.66 | 169.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 168.85 | 169.66 | 169.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 168.93 | 169.30 | 169.35 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 170.40 | 169.52 | 169.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 171.40 | 169.90 | 169.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 170.52 | 171.08 | 170.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 170.52 | 171.08 | 170.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 170.52 | 171.08 | 170.47 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 167.70 | 170.29 | 170.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 160.00 | 166.09 | 167.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 161.07 | 160.18 | 163.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 160.92 | 160.18 | 163.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 160.35 | 160.42 | 162.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 160.10 | 160.39 | 162.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:15:00 | 160.17 | 160.39 | 162.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 159.90 | 159.93 | 162.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 159.87 | 157.12 | 157.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 159.85 | 157.66 | 157.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:45:00 | 158.95 | 157.84 | 157.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 159.62 | 158.15 | 158.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 159.62 | 158.15 | 158.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 164.45 | 161.20 | 159.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 160.40 | 161.57 | 160.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 160.40 | 161.57 | 160.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 160.40 | 161.57 | 160.30 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 157.80 | 160.42 | 160.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 157.00 | 159.74 | 160.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 155.79 | 155.29 | 157.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 156.00 | 155.29 | 157.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 156.67 | 155.58 | 157.01 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 158.96 | 157.39 | 157.24 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 155.00 | 157.00 | 157.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 154.30 | 155.86 | 156.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 156.75 | 155.54 | 156.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 156.75 | 155.54 | 156.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 156.75 | 155.54 | 156.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 156.67 | 155.54 | 156.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 156.35 | 155.70 | 156.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 156.34 | 155.70 | 156.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 157.23 | 156.01 | 156.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 158.18 | 156.01 | 156.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 156.00 | 156.26 | 156.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 151.48 | 156.26 | 156.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 143.91 | 148.42 | 149.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 147.55 | 147.34 | 148.66 | SL hit (close>ema200) qty=0.50 sl=147.34 alert=retest2 |

### Cycle 225 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 148.85 | 147.20 | 147.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 149.20 | 147.60 | 147.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 148.23 | 148.36 | 147.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 15:15:00 | 148.00 | 148.30 | 147.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 148.00 | 148.30 | 147.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 152.68 | 148.30 | 147.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 167.95 | 165.27 | 162.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 166.52 | 167.76 | 167.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 165.58 | 167.32 | 167.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 165.52 | 165.22 | 166.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 165.52 | 165.22 | 166.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 165.52 | 165.22 | 166.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 165.73 | 165.22 | 166.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 165.93 | 165.37 | 166.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:30:00 | 165.28 | 165.36 | 165.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 165.33 | 165.36 | 165.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 166.13 | 165.49 | 165.75 | SL hit (close>static) qty=1.00 sl=166.12 alert=retest2 |

### Cycle 227 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 166.19 | 165.95 | 165.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 166.64 | 166.13 | 166.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 166.60 | 167.25 | 166.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 166.60 | 167.25 | 166.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 166.60 | 167.25 | 166.86 | EMA400 retest candle locked (from upside) |

### Cycle 228 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 165.45 | 166.51 | 166.63 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 167.65 | 166.74 | 166.72 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 165.98 | 166.79 | 166.84 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 168.50 | 166.94 | 166.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 169.12 | 167.62 | 167.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 167.08 | 169.06 | 168.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 167.08 | 169.06 | 168.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 167.08 | 169.06 | 168.53 | EMA400 retest candle locked (from upside) |

### Cycle 232 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 165.75 | 168.02 | 168.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 165.30 | 167.13 | 167.69 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 12:45:00 | 245.75 | 2023-05-19 09:15:00 | 233.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-15 13:30:00 | 245.78 | 2023-05-19 09:15:00 | 233.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-15 15:00:00 | 246.48 | 2023-05-19 09:15:00 | 234.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-15 12:45:00 | 245.75 | 2023-05-22 12:15:00 | 237.95 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2023-05-15 13:30:00 | 245.78 | 2023-05-22 12:15:00 | 237.95 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2023-05-15 15:00:00 | 246.48 | 2023-05-22 12:15:00 | 237.95 | STOP_HIT | 0.50 | 3.46% |
| BUY | retest2 | 2023-05-25 10:15:00 | 242.78 | 2023-05-30 14:15:00 | 239.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-06-16 09:15:00 | 234.25 | 2023-06-16 10:15:00 | 232.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-16 10:30:00 | 233.93 | 2023-06-16 11:15:00 | 232.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-06-26 09:30:00 | 240.93 | 2023-06-26 10:15:00 | 239.48 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-26 13:15:00 | 240.45 | 2023-06-27 11:15:00 | 238.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-06-30 14:15:00 | 236.50 | 2023-07-03 10:15:00 | 240.98 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-06-30 15:15:00 | 236.50 | 2023-07-03 10:15:00 | 240.98 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-07-11 13:15:00 | 243.40 | 2023-07-12 09:15:00 | 246.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-08-01 12:15:00 | 230.10 | 2023-08-07 11:15:00 | 227.58 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2023-08-01 14:45:00 | 230.10 | 2023-08-07 11:15:00 | 227.58 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2023-08-02 09:15:00 | 229.00 | 2023-08-07 11:15:00 | 227.58 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2023-08-08 13:15:00 | 228.45 | 2023-08-10 10:15:00 | 226.33 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-08-08 14:00:00 | 228.58 | 2023-08-10 10:15:00 | 226.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-08-09 11:15:00 | 228.48 | 2023-08-10 10:15:00 | 226.33 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest1 | 2023-08-17 12:45:00 | 216.90 | 2023-08-21 09:15:00 | 216.93 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest1 | 2023-08-17 15:00:00 | 216.50 | 2023-08-21 09:15:00 | 216.93 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-18 09:15:00 | 216.33 | 2023-08-21 09:15:00 | 216.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-08-18 10:30:00 | 215.20 | 2023-08-21 11:15:00 | 218.73 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2023-09-01 09:15:00 | 234.28 | 2023-09-04 09:15:00 | 230.55 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-09-01 14:30:00 | 235.00 | 2023-09-04 09:15:00 | 230.55 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-09-26 12:30:00 | 227.95 | 2023-10-03 09:15:00 | 230.75 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-09-27 10:00:00 | 228.23 | 2023-10-03 09:15:00 | 230.75 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-10-11 14:30:00 | 230.90 | 2023-10-19 09:15:00 | 230.75 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2023-10-11 15:15:00 | 231.50 | 2023-10-19 09:15:00 | 230.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-11-07 15:15:00 | 200.35 | 2023-11-08 09:15:00 | 196.93 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-11-10 09:15:00 | 196.80 | 2023-11-12 18:15:00 | 199.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-11-10 09:45:00 | 196.43 | 2023-11-12 18:15:00 | 199.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-11-10 10:15:00 | 196.90 | 2023-11-12 18:15:00 | 199.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-11-10 11:00:00 | 197.00 | 2023-11-12 18:15:00 | 199.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-11-13 09:15:00 | 197.15 | 2023-11-21 14:15:00 | 194.50 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2023-11-23 09:15:00 | 195.68 | 2023-11-28 14:15:00 | 194.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-11-23 10:30:00 | 195.03 | 2023-11-28 14:15:00 | 194.75 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-11-23 12:45:00 | 195.25 | 2023-11-28 14:15:00 | 194.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-11-23 14:45:00 | 195.10 | 2023-11-28 14:15:00 | 194.75 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-12-07 09:30:00 | 201.30 | 2023-12-08 13:15:00 | 199.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-12-19 11:45:00 | 203.55 | 2023-12-20 14:15:00 | 198.83 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2023-12-20 09:45:00 | 203.85 | 2023-12-20 14:15:00 | 198.83 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2023-12-29 11:45:00 | 209.38 | 2024-01-08 15:15:00 | 212.28 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2024-01-01 09:15:00 | 211.48 | 2024-01-08 15:15:00 | 212.28 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-01-02 11:15:00 | 209.40 | 2024-01-08 15:15:00 | 212.28 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2024-01-16 14:45:00 | 216.78 | 2024-01-18 09:15:00 | 213.35 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-01-17 11:45:00 | 216.75 | 2024-01-18 09:15:00 | 213.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-01-17 12:15:00 | 217.40 | 2024-01-18 09:15:00 | 213.35 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-01-17 12:45:00 | 216.95 | 2024-01-18 09:15:00 | 213.35 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-01-18 11:15:00 | 217.03 | 2024-01-23 15:15:00 | 217.05 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-01-18 11:45:00 | 217.13 | 2024-01-23 15:15:00 | 217.05 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-01-18 12:45:00 | 216.55 | 2024-01-23 15:15:00 | 217.05 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-02-06 09:15:00 | 222.15 | 2024-02-08 13:15:00 | 221.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-02-12 10:15:00 | 216.60 | 2024-02-15 09:15:00 | 221.78 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-02-13 11:15:00 | 215.83 | 2024-02-15 09:15:00 | 221.78 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-02-13 14:30:00 | 216.60 | 2024-02-15 09:15:00 | 221.78 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-02-14 09:15:00 | 214.28 | 2024-02-15 09:15:00 | 221.78 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-02-21 10:15:00 | 218.93 | 2024-02-22 11:15:00 | 220.45 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-02-21 14:00:00 | 219.00 | 2024-02-22 11:15:00 | 220.45 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-02-21 15:15:00 | 218.38 | 2024-02-22 11:15:00 | 220.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-02-22 09:45:00 | 218.08 | 2024-02-22 11:15:00 | 220.45 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-03-13 09:15:00 | 210.10 | 2024-03-20 09:15:00 | 199.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 09:15:00 | 210.10 | 2024-03-20 13:15:00 | 202.25 | STOP_HIT | 0.50 | 3.74% |
| BUY | retest1 | 2024-04-25 09:15:00 | 227.10 | 2024-04-25 12:15:00 | 223.75 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest1 | 2024-04-25 11:45:00 | 226.90 | 2024-04-25 12:15:00 | 223.75 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-05-03 09:15:00 | 232.25 | 2024-05-03 10:15:00 | 229.83 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-05-15 10:00:00 | 220.05 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-05-16 09:45:00 | 219.55 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-05-16 10:45:00 | 220.15 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-05-16 12:00:00 | 220.08 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-05-16 13:15:00 | 219.00 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-16 14:30:00 | 219.15 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-05-17 09:30:00 | 219.15 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-05-17 13:30:00 | 219.15 | 2024-05-17 15:15:00 | 221.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-05-23 09:15:00 | 223.35 | 2024-05-30 09:15:00 | 226.60 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2024-06-11 13:30:00 | 237.18 | 2024-06-19 09:15:00 | 236.95 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-06-12 09:15:00 | 237.60 | 2024-06-19 09:15:00 | 236.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-06-20 13:45:00 | 236.70 | 2024-06-20 14:15:00 | 238.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-06-21 12:30:00 | 235.85 | 2024-06-25 09:15:00 | 239.78 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-06-21 14:00:00 | 236.50 | 2024-06-25 09:15:00 | 239.78 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-06-24 09:30:00 | 236.25 | 2024-06-25 09:15:00 | 239.78 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-06-24 13:45:00 | 237.40 | 2024-06-25 09:15:00 | 239.78 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-04 11:15:00 | 259.20 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2024-07-04 11:45:00 | 259.23 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2024-07-04 13:15:00 | 259.58 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-07-04 15:15:00 | 259.55 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-07-05 12:30:00 | 260.30 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-07-05 13:00:00 | 260.45 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-07-05 14:45:00 | 262.05 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-07-08 14:30:00 | 261.75 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-07-09 11:30:00 | 261.50 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-07-10 09:15:00 | 268.02 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-07-11 10:00:00 | 261.90 | 2024-07-19 09:15:00 | 263.50 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-08-01 09:30:00 | 274.88 | 2024-08-01 11:15:00 | 272.77 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-08-16 10:15:00 | 269.45 | 2024-08-16 11:15:00 | 272.05 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-08-20 15:15:00 | 274.85 | 2024-08-22 13:15:00 | 271.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-08-21 09:45:00 | 276.27 | 2024-08-22 13:15:00 | 271.55 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-08-27 11:30:00 | 265.00 | 2024-08-28 09:15:00 | 269.45 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-09-03 09:15:00 | 275.60 | 2024-09-06 09:15:00 | 271.77 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-09-03 10:45:00 | 274.58 | 2024-09-06 09:15:00 | 271.77 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-03 12:00:00 | 275.00 | 2024-09-06 09:15:00 | 271.77 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-09-10 14:45:00 | 269.05 | 2024-09-16 14:15:00 | 264.68 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2024-09-11 09:15:00 | 268.00 | 2024-09-16 14:15:00 | 264.68 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-10-08 11:30:00 | 271.83 | 2024-10-14 12:15:00 | 258.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-08 11:30:00 | 271.83 | 2024-10-15 09:15:00 | 262.50 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest2 | 2024-11-08 15:15:00 | 221.75 | 2024-11-12 09:15:00 | 214.10 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-11-11 09:30:00 | 221.58 | 2024-11-12 09:15:00 | 214.10 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-11-11 10:15:00 | 221.53 | 2024-11-12 09:15:00 | 214.10 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2024-11-26 09:45:00 | 160.45 | 2024-11-29 14:15:00 | 163.78 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-11-26 11:45:00 | 160.82 | 2024-11-29 14:15:00 | 163.78 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-26 14:00:00 | 160.95 | 2024-11-29 14:15:00 | 163.78 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-11-28 09:30:00 | 161.00 | 2024-11-29 14:15:00 | 163.78 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-29 09:45:00 | 158.93 | 2024-11-29 14:15:00 | 163.78 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-12-11 14:15:00 | 194.58 | 2024-12-17 11:15:00 | 192.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-12 11:45:00 | 193.90 | 2024-12-17 11:15:00 | 192.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-12 12:45:00 | 194.35 | 2024-12-17 12:15:00 | 192.05 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-12-13 11:15:00 | 194.63 | 2024-12-17 12:15:00 | 192.05 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-13 15:15:00 | 196.35 | 2024-12-17 12:15:00 | 192.05 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-12-16 09:45:00 | 196.98 | 2024-12-17 12:15:00 | 192.05 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-01-07 09:30:00 | 214.73 | 2025-01-07 10:15:00 | 210.53 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-01-09 11:15:00 | 212.60 | 2025-01-09 12:15:00 | 214.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-01-09 12:15:00 | 212.78 | 2025-01-09 12:15:00 | 214.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-01-14 13:15:00 | 203.45 | 2025-01-22 09:15:00 | 193.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 09:45:00 | 203.03 | 2025-01-22 09:15:00 | 192.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 11:00:00 | 203.40 | 2025-01-22 09:15:00 | 193.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 12:30:00 | 203.50 | 2025-01-22 09:15:00 | 193.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:00:00 | 202.08 | 2025-01-22 09:15:00 | 192.23 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-01-17 14:00:00 | 202.35 | 2025-01-22 09:15:00 | 192.26 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-01-17 15:00:00 | 202.38 | 2025-01-22 10:15:00 | 191.98 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-01-14 13:15:00 | 203.45 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2025-01-15 09:45:00 | 203.03 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2025-01-15 11:00:00 | 203.40 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2025-01-15 12:30:00 | 203.50 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2025-01-16 11:00:00 | 202.08 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-01-17 14:00:00 | 202.35 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-01-17 15:00:00 | 202.38 | 2025-01-22 14:15:00 | 194.18 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-01-20 09:15:00 | 198.28 | 2025-01-27 09:15:00 | 188.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 14:30:00 | 197.55 | 2025-01-27 09:15:00 | 187.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 198.28 | 2025-01-28 11:15:00 | 189.78 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-01-21 14:30:00 | 197.55 | 2025-01-28 11:15:00 | 189.78 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2025-02-14 10:15:00 | 187.81 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-02-14 14:45:00 | 187.30 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-02-17 12:15:00 | 187.96 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-02-18 09:15:00 | 187.37 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-02-18 11:30:00 | 187.58 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-02-18 12:15:00 | 187.53 | 2025-02-18 14:15:00 | 192.63 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-02-28 09:15:00 | 191.79 | 2025-03-03 11:15:00 | 182.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 191.79 | 2025-03-03 12:15:00 | 187.97 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2025-03-27 13:15:00 | 196.71 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-03-27 13:45:00 | 196.12 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-04-03 09:15:00 | 204.38 | 2025-04-04 11:15:00 | 198.28 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-04-16 09:15:00 | 174.33 | 2025-04-21 09:15:00 | 180.46 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-04-16 13:15:00 | 175.23 | 2025-04-21 09:15:00 | 180.46 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-04-17 09:15:00 | 175.75 | 2025-04-21 09:15:00 | 180.46 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-04-17 09:45:00 | 175.58 | 2025-04-21 09:15:00 | 180.46 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-04-23 15:00:00 | 185.67 | 2025-04-25 09:15:00 | 181.96 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-04-24 09:15:00 | 185.58 | 2025-04-25 09:15:00 | 181.96 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-04-24 11:30:00 | 186.85 | 2025-04-25 09:15:00 | 181.96 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-04-24 15:15:00 | 185.75 | 2025-04-25 09:15:00 | 181.96 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-04-30 10:45:00 | 191.09 | 2025-05-08 15:15:00 | 198.80 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-05-14 09:15:00 | 204.19 | 2025-05-15 12:15:00 | 202.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-15 10:30:00 | 203.90 | 2025-05-15 12:15:00 | 202.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-05-20 09:15:00 | 212.99 | 2025-05-20 11:15:00 | 209.35 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest1 | 2025-05-20 09:45:00 | 212.30 | 2025-05-20 11:15:00 | 209.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-02 11:15:00 | 209.40 | 2025-06-04 15:15:00 | 208.93 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-20 13:45:00 | 205.07 | 2025-06-23 09:15:00 | 209.28 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-07-04 10:30:00 | 223.00 | 2025-07-09 13:15:00 | 223.57 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-07-17 11:30:00 | 220.91 | 2025-07-17 14:15:00 | 219.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-17 13:30:00 | 220.84 | 2025-07-17 14:15:00 | 219.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-23 10:30:00 | 212.22 | 2025-07-25 12:15:00 | 201.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 212.22 | 2025-07-28 09:15:00 | 206.06 | STOP_HIT | 0.50 | 2.90% |
| BUY | retest2 | 2025-08-06 12:15:00 | 205.25 | 2025-08-06 15:15:00 | 203.81 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-08 13:00:00 | 202.90 | 2025-08-12 09:15:00 | 207.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-08 14:45:00 | 202.40 | 2025-08-12 09:15:00 | 207.25 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-08-11 14:45:00 | 202.87 | 2025-08-12 09:15:00 | 207.25 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-22 11:15:00 | 207.29 | 2025-08-28 10:15:00 | 208.33 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-08-28 09:45:00 | 207.99 | 2025-08-28 10:15:00 | 208.33 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-08-29 12:45:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-29 13:30:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-16 09:15:00 | 215.59 | 2025-09-18 12:15:00 | 214.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-16 10:15:00 | 215.81 | 2025-09-18 12:15:00 | 214.74 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-16 12:15:00 | 215.55 | 2025-09-22 09:15:00 | 214.52 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-16 13:45:00 | 215.55 | 2025-09-22 09:15:00 | 214.52 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-18 09:15:00 | 215.56 | 2025-09-22 10:15:00 | 214.55 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-09-18 10:15:00 | 215.64 | 2025-09-22 10:15:00 | 214.55 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-18 15:00:00 | 215.75 | 2025-09-22 10:15:00 | 214.55 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-19 10:15:00 | 215.89 | 2025-09-22 10:15:00 | 214.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-17 10:15:00 | 211.12 | 2025-10-23 12:15:00 | 213.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-17 11:00:00 | 210.60 | 2025-10-23 12:15:00 | 213.70 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-10-17 11:45:00 | 210.58 | 2025-10-23 12:15:00 | 213.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-27 10:30:00 | 213.20 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-28 10:30:00 | 212.50 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-04 09:15:00 | 218.06 | 2025-11-06 09:15:00 | 211.19 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-11-10 09:30:00 | 211.38 | 2025-11-13 09:15:00 | 215.05 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-11-10 11:00:00 | 211.29 | 2025-11-13 09:15:00 | 215.05 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-11 11:00:00 | 209.39 | 2025-11-13 09:15:00 | 215.05 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-11-24 11:30:00 | 204.75 | 2025-11-25 12:15:00 | 194.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 11:30:00 | 204.75 | 2025-11-26 09:15:00 | 197.15 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-12-10 11:15:00 | 185.30 | 2025-12-12 14:15:00 | 186.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-10 12:30:00 | 185.15 | 2025-12-12 14:15:00 | 186.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-15 13:15:00 | 185.95 | 2025-12-15 14:15:00 | 185.51 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-12-22 09:15:00 | 196.25 | 2025-12-24 13:15:00 | 193.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-12 11:15:00 | 182.20 | 2026-01-22 14:15:00 | 179.32 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2026-01-12 12:45:00 | 183.24 | 2026-01-22 14:15:00 | 179.32 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2026-01-12 13:15:00 | 183.06 | 2026-01-22 14:15:00 | 179.32 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2026-01-13 09:30:00 | 183.20 | 2026-01-22 14:15:00 | 179.32 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2026-01-14 13:30:00 | 181.20 | 2026-01-22 14:15:00 | 179.32 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2026-02-03 10:15:00 | 172.70 | 2026-02-06 10:15:00 | 174.24 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-02-11 13:15:00 | 176.24 | 2026-02-12 09:15:00 | 169.32 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2026-02-11 14:30:00 | 176.49 | 2026-02-12 09:15:00 | 169.32 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2026-03-05 12:45:00 | 160.10 | 2026-03-11 09:15:00 | 159.62 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2026-03-05 13:15:00 | 160.17 | 2026-03-11 09:15:00 | 159.62 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2026-03-05 15:00:00 | 159.90 | 2026-03-11 09:15:00 | 159.62 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-03-10 11:15:00 | 159.87 | 2026-03-11 09:15:00 | 159.62 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2026-03-10 12:45:00 | 158.95 | 2026-03-11 09:15:00 | 159.62 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-03-23 09:15:00 | 151.48 | 2026-03-30 09:15:00 | 143.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 151.48 | 2026-03-30 12:15:00 | 147.55 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2026-04-08 09:15:00 | 152.68 | 2026-04-16 09:15:00 | 167.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:30:00 | 165.28 | 2026-04-28 10:15:00 | 166.13 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-04-27 12:00:00 | 165.33 | 2026-04-28 10:15:00 | 166.13 | STOP_HIT | 1.00 | -0.48% |
