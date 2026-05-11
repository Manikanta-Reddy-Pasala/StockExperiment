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
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 118 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 113 |
| PARTIAL | 18 |
| TARGET_HIT | 13 |
| STOP_HIT | 104 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 96
- **Target hits / Stop hits / Partials:** 13 / 104 / 18
- **Avg / median % per leg:** 0.03% / -1.32%
- **Sum % (uncompounded):** 4.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 5 | 10.4% | 5 | 43 | 0 | -0.93% | -44.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 48 | 5 | 10.4% | 5 | 43 | 0 | -0.93% | -44.8% |
| SELL (all) | 87 | 34 | 39.1% | 8 | 61 | 18 | 0.57% | 49.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.92% | -31.7% |
| SELL @ 3rd Alert (retest2) | 83 | 34 | 41.0% | 8 | 57 | 18 | 0.98% | 81.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -7.92% | -31.7% |
| retest2 (combined) | 131 | 39 | 29.8% | 13 | 100 | 18 | 0.28% | 36.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 248.55 | 241.41 | 241.40 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 11:15:00 | 235.50 | 241.41 | 241.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 12:15:00 | 235.00 | 241.35 | 241.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 10:15:00 | 230.60 | 226.54 | 231.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 11:00:00 | 230.60 | 226.54 | 231.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 231.33 | 226.66 | 231.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 14:00:00 | 231.33 | 226.66 | 231.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 231.15 | 226.71 | 231.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 14:30:00 | 231.43 | 226.71 | 231.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 233.10 | 226.81 | 231.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:00:00 | 233.10 | 226.81 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 232.93 | 226.88 | 231.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 11:00:00 | 232.93 | 226.88 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 232.50 | 227.04 | 231.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:45:00 | 232.55 | 227.04 | 231.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 231.50 | 227.13 | 231.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 232.73 | 227.13 | 231.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 234.80 | 227.21 | 231.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 234.25 | 227.21 | 231.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 235.00 | 227.28 | 231.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 12:30:00 | 234.60 | 227.43 | 231.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 13:00:00 | 234.45 | 227.43 | 231.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 14:45:00 | 234.18 | 227.91 | 231.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 09:45:00 | 234.38 | 228.39 | 231.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 234.45 | 228.45 | 231.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 12:45:00 | 233.15 | 228.55 | 231.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 14:45:00 | 233.50 | 228.64 | 231.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 12:15:00 | 237.50 | 228.87 | 231.74 | SL hit (close>static) qty=1.00 sl=237.43 alert=retest2 |

### Cycle 3 — BUY (started 2023-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 15:15:00 | 243.50 | 232.30 | 232.26 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 206.40 | 232.13 | 232.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 204.43 | 231.59 | 231.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 10:15:00 | 201.85 | 200.67 | 209.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-04 10:45:00 | 201.45 | 200.67 | 209.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 206.75 | 200.97 | 206.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 206.75 | 200.97 | 206.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 206.00 | 201.02 | 206.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 13:00:00 | 204.78 | 201.06 | 206.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 11:30:00 | 205.30 | 201.33 | 206.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 09:30:00 | 205.13 | 201.55 | 206.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 10:45:00 | 205.03 | 201.60 | 206.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 11:15:00 | 206.23 | 201.64 | 206.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:00:00 | 206.23 | 201.64 | 206.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 205.48 | 201.68 | 206.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:45:00 | 206.25 | 201.68 | 206.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 205.88 | 201.72 | 206.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 205.93 | 201.72 | 206.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 205.50 | 201.76 | 206.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 15:15:00 | 205.15 | 201.76 | 206.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 09:15:00 | 206.35 | 201.84 | 206.09 | SL hit (close>static) qty=1.00 sl=206.13 alert=retest2 |

### Cycle 5 — BUY (started 2024-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 15:15:00 | 217.00 | 208.55 | 208.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-17 13:15:00 | 217.35 | 208.94 | 208.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 10:15:00 | 211.50 | 212.26 | 210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 10:15:00 | 211.50 | 212.26 | 210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 211.50 | 212.26 | 210.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:45:00 | 210.58 | 212.26 | 210.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 210.65 | 212.24 | 210.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:45:00 | 210.75 | 212.24 | 210.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 205.35 | 212.17 | 210.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 205.35 | 212.17 | 210.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 201.85 | 212.07 | 210.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 201.85 | 212.07 | 210.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 209.90 | 211.80 | 210.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 14:30:00 | 209.05 | 211.80 | 210.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 213.43 | 216.03 | 213.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 213.43 | 216.03 | 213.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 213.78 | 216.01 | 213.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 211.73 | 216.01 | 213.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 216.38 | 216.02 | 213.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:30:00 | 213.18 | 216.02 | 213.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 215.88 | 216.06 | 213.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 13:30:00 | 217.30 | 216.05 | 213.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 14:30:00 | 217.60 | 216.07 | 213.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:45:00 | 217.68 | 217.07 | 214.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 11:00:00 | 217.70 | 217.73 | 215.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 215.20 | 217.68 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 214.73 | 217.68 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 215.40 | 217.65 | 215.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:30:00 | 215.13 | 217.65 | 215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 214.23 | 217.60 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:00:00 | 214.23 | 217.60 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 214.18 | 217.57 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:00:00 | 214.18 | 217.57 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 212.45 | 217.51 | 215.06 | SL hit (close<static) qty=1.00 sl=213.35 alert=retest2 |

### Cycle 6 — SELL (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 13:15:00 | 204.75 | 213.99 | 214.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 202.10 | 213.69 | 213.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 213.13 | 211.14 | 212.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 10:15:00 | 213.13 | 211.14 | 212.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 213.13 | 211.14 | 212.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:00:00 | 213.13 | 211.14 | 212.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 11:15:00 | 213.33 | 211.16 | 212.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:00:00 | 213.33 | 211.16 | 212.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 213.25 | 211.18 | 212.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:30:00 | 213.45 | 211.18 | 212.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 214.13 | 211.21 | 212.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:30:00 | 213.30 | 211.21 | 212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 212.63 | 211.24 | 212.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 214.20 | 211.24 | 212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 213.78 | 211.26 | 212.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:30:00 | 214.90 | 211.26 | 212.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 09:15:00 | 219.70 | 213.44 | 213.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 10:15:00 | 225.45 | 213.91 | 213.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 12:15:00 | 221.18 | 222.11 | 218.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-18 12:45:00 | 221.58 | 222.11 | 218.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 218.03 | 222.06 | 218.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 218.03 | 222.06 | 218.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 220.45 | 222.04 | 218.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 223.25 | 222.04 | 218.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 220.05 | 222.02 | 218.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:00:00 | 225.23 | 221.63 | 218.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:15:00 | 224.80 | 221.70 | 218.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 226.00 | 221.78 | 218.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:30:00 | 226.35 | 222.34 | 219.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 221.55 | 224.95 | 221.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 15:00:00 | 221.55 | 224.95 | 221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 221.70 | 224.92 | 221.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 222.35 | 224.92 | 221.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 223.90 | 224.91 | 221.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 229.43 | 224.61 | 221.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 226.10 | 224.82 | 221.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:45:00 | 226.90 | 224.64 | 221.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 218.00 | 224.58 | 221.51 | SL hit (close<static) qty=1.00 sl=219.65 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 223.25 | 264.71 | 264.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 221.00 | 263.04 | 263.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 192.80 | 191.89 | 213.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 190.83 | 191.92 | 212.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:45:00 | 191.15 | 192.16 | 211.25 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 190.90 | 192.33 | 210.14 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 14:00:00 | 191.23 | 192.91 | 208.91 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 205.53 | 194.66 | 205.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 205.53 | 194.66 | 205.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 206.15 | 194.77 | 205.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 206.15 | 194.77 | 205.72 | SL hit (close>ema400) qty=1.00 sl=205.72 alert=retest1 |

### Cycle 9 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 204.20 | 193.51 | 193.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 206.60 | 194.22 | 193.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 205.17 | 205.97 | 201.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 205.17 | 205.97 | 201.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 196.50 | 205.80 | 201.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:30:00 | 194.72 | 205.80 | 201.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 202.06 | 205.74 | 202.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 202.06 | 205.74 | 202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 202.46 | 205.71 | 202.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 201.98 | 205.71 | 202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 202.21 | 205.67 | 202.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:30:00 | 202.40 | 205.67 | 202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 202.18 | 205.64 | 202.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 204.34 | 205.64 | 202.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 205.26 | 205.63 | 202.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 208.01 | 205.63 | 202.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 207.20 | 205.68 | 202.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 207.10 | 205.69 | 202.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 207.43 | 205.70 | 202.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 12:15:00 | 228.81 | 210.06 | 205.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 204.96 | 208.06 | 208.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 204.53 | 208.03 | 208.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 207.36 | 207.32 | 207.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 207.94 | 207.33 | 207.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 207.94 | 207.33 | 207.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 206.41 | 207.32 | 207.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 206.34 | 207.32 | 207.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 206.26 | 207.30 | 207.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 208.08 | 207.30 | 207.64 | SL hit (close>static) qty=1.00 sl=207.99 alert=retest2 |

### Cycle 11 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 217.03 | 207.92 | 207.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 217.52 | 208.10 | 208.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 209.80 | 210.15 | 209.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 209.80 | 210.15 | 209.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 208.93 | 210.15 | 209.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 208.93 | 210.15 | 209.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 209.20 | 210.14 | 209.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 209.82 | 210.14 | 209.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 208.65 | 212.05 | 210.58 | SL hit (close<static) qty=1.00 sl=208.91 alert=retest2 |

### Cycle 12 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 203.78 | 210.90 | 210.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 202.21 | 210.81 | 210.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 196.90 | 196.22 | 201.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 196.90 | 196.22 | 201.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 164.45 | 157.03 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 164.68 | 157.03 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 164.71 | 157.10 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 165.22 | 157.10 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 165.34 | 157.19 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:00:00 | 165.34 | 157.19 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 165.74 | 157.27 | 164.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:30:00 | 165.78 | 157.27 | 164.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 164.91 | 157.91 | 164.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:30:00 | 164.99 | 157.91 | 164.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 165.65 | 157.99 | 164.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 165.65 | 157.99 | 164.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 168.79 | 159.49 | 164.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 166.15 | 160.91 | 165.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 166.24 | 161.57 | 165.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:00:00 | 166.30 | 161.71 | 165.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 166.19 | 161.76 | 165.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 166.13 | 162.34 | 165.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:15:00 | 167.21 | 162.34 | 165.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 167.65 | 162.59 | 165.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 168.19 | 162.59 | 165.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 169.99 | 163.45 | 165.50 | SL hit (close>static) qty=1.00 sl=169.77 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-31 12:30:00 | 234.60 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-08-31 13:00:00 | 234.45 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-09-01 14:45:00 | 234.18 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-09-06 09:45:00 | 234.38 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-09-06 12:45:00 | 233.15 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2023-09-06 14:45:00 | 233.50 | 2023-09-07 12:15:00 | 237.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-09-12 10:00:00 | 233.48 | 2023-09-12 10:15:00 | 235.68 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-09-12 12:00:00 | 233.33 | 2023-09-14 14:15:00 | 235.55 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-09-21 12:45:00 | 229.65 | 2023-10-03 10:15:00 | 233.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-09-21 13:45:00 | 229.75 | 2023-10-03 10:15:00 | 233.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-10-04 11:45:00 | 229.55 | 2023-10-12 09:15:00 | 233.85 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2023-10-04 12:45:00 | 228.05 | 2023-10-12 09:15:00 | 233.85 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2023-10-10 11:15:00 | 228.30 | 2023-10-12 09:15:00 | 233.85 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2023-12-20 13:00:00 | 204.78 | 2023-12-28 09:15:00 | 206.35 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-12-26 11:30:00 | 205.30 | 2023-12-28 12:15:00 | 207.45 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-12-27 09:30:00 | 205.13 | 2023-12-28 12:15:00 | 207.45 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-12-27 10:45:00 | 205.03 | 2023-12-28 12:15:00 | 207.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-12-27 15:15:00 | 205.15 | 2023-12-28 12:15:00 | 207.45 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-01-10 09:15:00 | 204.90 | 2024-01-10 09:15:00 | 207.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-02-14 13:30:00 | 217.30 | 2024-02-27 13:15:00 | 212.45 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-02-14 14:30:00 | 217.60 | 2024-02-27 13:15:00 | 212.45 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-02-20 10:45:00 | 217.68 | 2024-02-27 13:15:00 | 212.45 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-02-26 11:00:00 | 217.70 | 2024-02-27 13:15:00 | 212.45 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-04-23 10:00:00 | 225.23 | 2024-05-13 09:15:00 | 218.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-04-23 13:15:00 | 224.80 | 2024-05-13 09:15:00 | 218.00 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-04-24 09:15:00 | 226.00 | 2024-05-13 09:15:00 | 218.00 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-04-25 14:30:00 | 226.35 | 2024-05-14 09:15:00 | 215.93 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2024-05-08 09:15:00 | 229.43 | 2024-05-14 09:15:00 | 215.93 | STOP_HIT | 1.00 | -5.88% |
| BUY | retest2 | 2024-05-08 15:15:00 | 226.10 | 2024-05-14 09:15:00 | 215.93 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2024-05-10 09:45:00 | 226.90 | 2024-05-14 09:15:00 | 215.93 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-05-23 13:15:00 | 226.70 | 2024-06-04 11:15:00 | 215.80 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-06-03 09:15:00 | 230.95 | 2024-06-04 11:15:00 | 215.80 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest2 | 2024-06-06 09:15:00 | 230.75 | 2024-06-28 11:15:00 | 253.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-18 10:30:00 | 225.23 | 2024-10-21 09:15:00 | 223.25 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-10-18 13:00:00 | 225.08 | 2024-10-21 09:15:00 | 223.25 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest1 | 2024-12-09 09:30:00 | 190.83 | 2024-12-31 11:15:00 | 206.15 | STOP_HIT | 1.00 | -8.03% |
| SELL | retest1 | 2024-12-11 10:45:00 | 191.15 | 2024-12-31 11:15:00 | 206.15 | STOP_HIT | 1.00 | -7.85% |
| SELL | retest1 | 2024-12-13 10:15:00 | 190.90 | 2024-12-31 11:15:00 | 206.15 | STOP_HIT | 1.00 | -7.99% |
| SELL | retest1 | 2024-12-17 14:00:00 | 191.23 | 2024-12-31 11:15:00 | 206.15 | STOP_HIT | 1.00 | -7.80% |
| SELL | retest2 | 2025-01-08 12:15:00 | 204.75 | 2025-01-09 09:15:00 | 215.98 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2025-01-08 13:00:00 | 204.15 | 2025-01-09 09:15:00 | 215.98 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2025-01-13 10:00:00 | 204.60 | 2025-01-14 09:15:00 | 206.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-13 10:45:00 | 203.80 | 2025-01-14 09:15:00 | 206.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-14 13:15:00 | 203.45 | 2025-01-22 09:15:00 | 193.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 09:45:00 | 203.03 | 2025-01-22 09:15:00 | 192.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 11:00:00 | 203.40 | 2025-01-22 09:15:00 | 193.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 12:30:00 | 203.50 | 2025-01-22 09:15:00 | 193.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:00:00 | 202.08 | 2025-01-22 09:15:00 | 192.23 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-01-17 14:00:00 | 202.35 | 2025-01-22 09:15:00 | 192.26 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-01-17 15:00:00 | 202.38 | 2025-01-22 10:15:00 | 191.98 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-01-20 09:15:00 | 198.28 | 2025-01-27 09:15:00 | 188.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 13:15:00 | 203.45 | 2025-01-28 09:15:00 | 183.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 09:45:00 | 203.03 | 2025-01-28 09:15:00 | 182.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 11:00:00 | 203.40 | 2025-01-28 09:15:00 | 183.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 12:30:00 | 203.50 | 2025-01-28 09:15:00 | 183.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-16 11:00:00 | 202.08 | 2025-01-28 09:15:00 | 181.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-17 14:00:00 | 202.35 | 2025-01-28 09:15:00 | 182.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-17 15:00:00 | 202.38 | 2025-01-28 09:15:00 | 182.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 198.28 | 2025-01-29 13:15:00 | 199.15 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2025-02-01 11:45:00 | 200.07 | 2025-02-04 10:15:00 | 190.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 200.07 | 2025-02-05 09:15:00 | 204.72 | STOP_HIT | 0.50 | -2.32% |
| SELL | retest2 | 2025-02-06 12:15:00 | 201.29 | 2025-02-11 09:15:00 | 191.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 13:45:00 | 200.85 | 2025-02-11 09:15:00 | 190.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 201.01 | 2025-02-11 09:15:00 | 190.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 199.23 | 2025-02-11 12:15:00 | 190.17 | PARTIAL | 0.50 | 4.55% |
| SELL | retest2 | 2025-02-07 12:00:00 | 200.18 | 2025-02-11 12:15:00 | 190.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 200.17 | 2025-02-11 12:15:00 | 189.94 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-02-07 13:30:00 | 199.94 | 2025-02-12 09:15:00 | 189.27 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-02-06 12:15:00 | 201.29 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2025-02-06 13:45:00 | 200.85 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2025-02-07 09:15:00 | 201.01 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-02-07 10:15:00 | 199.23 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-02-07 12:00:00 | 200.18 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2025-02-07 12:45:00 | 200.17 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-02-07 13:30:00 | 199.94 | 2025-02-20 09:15:00 | 197.70 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2025-02-24 12:45:00 | 198.08 | 2025-02-28 11:15:00 | 188.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 12:45:00 | 198.08 | 2025-03-04 09:15:00 | 178.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-20 10:00:00 | 197.92 | 2025-03-24 09:15:00 | 205.50 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-03-20 14:15:00 | 198.45 | 2025-03-24 09:15:00 | 205.50 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-03-21 10:00:00 | 197.54 | 2025-03-24 09:15:00 | 205.50 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-03-26 13:30:00 | 194.69 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-03-26 15:00:00 | 195.08 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-03-27 10:00:00 | 195.90 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-03-27 14:00:00 | 195.56 | 2025-03-28 09:15:00 | 202.08 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-04-01 11:00:00 | 200.94 | 2025-04-03 10:15:00 | 203.78 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-04-01 11:30:00 | 200.57 | 2025-04-03 10:15:00 | 203.78 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-04-01 13:00:00 | 200.70 | 2025-04-03 10:15:00 | 203.78 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-04-02 09:15:00 | 200.12 | 2025-04-03 10:15:00 | 203.78 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-07 09:15:00 | 190.44 | 2025-04-08 14:15:00 | 180.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 190.44 | 2025-04-22 13:15:00 | 189.20 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2025-05-02 10:15:00 | 196.52 | 2025-05-05 09:15:00 | 200.75 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-05-02 11:30:00 | 196.64 | 2025-05-05 09:15:00 | 200.75 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-02 12:15:00 | 195.37 | 2025-05-05 09:15:00 | 200.75 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-06-23 09:15:00 | 208.01 | 2025-07-04 12:15:00 | 228.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 11:00:00 | 207.20 | 2025-07-04 12:15:00 | 227.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 13:15:00 | 207.10 | 2025-07-04 12:15:00 | 227.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 13:45:00 | 207.43 | 2025-07-04 12:15:00 | 228.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 206.34 | 2025-08-22 11:15:00 | 208.08 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-22 09:45:00 | 206.26 | 2025-08-22 11:15:00 | 208.08 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-28 09:15:00 | 206.19 | 2025-08-28 10:15:00 | 208.33 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-28 15:15:00 | 206.20 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-08-29 12:45:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-29 13:30:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-10 09:15:00 | 209.82 | 2025-09-23 14:15:00 | 208.65 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-30 14:45:00 | 209.33 | 2025-09-30 15:15:00 | 208.17 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-01 09:45:00 | 210.60 | 2025-10-01 11:15:00 | 208.37 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-03 15:15:00 | 209.51 | 2025-10-06 09:15:00 | 208.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-15 11:00:00 | 212.55 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-10-15 15:15:00 | 212.69 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-10-16 10:00:00 | 212.24 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-23 12:15:00 | 212.26 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-27 10:30:00 | 213.20 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-28 10:30:00 | 212.50 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-29 09:15:00 | 213.01 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-29 09:45:00 | 212.57 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-29 15:15:00 | 213.00 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-31 09:30:00 | 212.70 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-31 12:00:00 | 213.07 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-31 13:45:00 | 212.69 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-11-03 14:45:00 | 215.38 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-11-04 12:00:00 | 214.98 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-11-04 13:15:00 | 215.24 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-04 14:45:00 | 214.76 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-11-07 12:45:00 | 213.00 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-11-07 14:15:00 | 212.50 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-07 15:15:00 | 212.44 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-11 10:45:00 | 215.91 | 2025-11-11 11:15:00 | 209.89 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-11-11 12:30:00 | 211.07 | 2025-11-11 14:15:00 | 208.29 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-12 13:45:00 | 211.23 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-13 09:15:00 | 214.51 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-11-17 13:45:00 | 211.07 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-23 14:15:00 | 166.15 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-28 09:45:00 | 166.24 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-04-28 13:00:00 | 166.30 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-28 14:00:00 | 166.19 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.29% |
