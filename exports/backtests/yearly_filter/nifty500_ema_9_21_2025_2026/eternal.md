# Eternal Ltd. (ETERNAL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 256.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 52 |
| ALERT2 | 52 |
| ALERT2_SKIP | 28 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 15 / 37
- **Target hits / Stop hits / Partials:** 5 / 42 / 5
- **Avg / median % per leg:** 0.77% / -0.63%
- **Sum % (uncompounded):** 39.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 5 | 29.4% | 4 | 13 | 0 | 1.28% | 21.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 5 | 29.4% | 4 | 13 | 0 | 1.28% | 21.7% |
| SELL (all) | 35 | 10 | 28.6% | 1 | 29 | 5 | 0.52% | 18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 10 | 28.6% | 1 | 29 | 5 | 0.52% | 18.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 15 | 28.8% | 5 | 42 | 5 | 0.77% | 40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 236.91 | 231.58 | 231.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 239.70 | 236.81 | 235.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 243.10 | 243.71 | 241.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 242.80 | 243.71 | 241.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 241.30 | 243.23 | 241.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 241.30 | 243.23 | 241.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 240.92 | 242.77 | 241.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:15:00 | 241.21 | 242.77 | 241.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 240.90 | 242.39 | 241.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:15:00 | 242.48 | 242.39 | 241.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 240.13 | 241.94 | 241.02 | SL hit (close<static) qty=1.00 sl=240.51 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 234.73 | 239.41 | 240.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 232.15 | 237.96 | 239.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 230.01 | 229.94 | 232.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 230.01 | 229.94 | 232.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 235.02 | 230.36 | 231.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 235.02 | 230.36 | 231.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 236.15 | 231.52 | 231.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 236.15 | 231.52 | 231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 237.58 | 232.73 | 232.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 239.32 | 234.05 | 232.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 231.00 | 234.99 | 233.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 231.00 | 234.99 | 233.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 231.00 | 234.99 | 233.82 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 227.81 | 232.68 | 232.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 226.03 | 231.35 | 232.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 227.20 | 225.10 | 226.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 227.20 | 225.10 | 226.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 227.20 | 225.10 | 226.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 226.09 | 225.10 | 226.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 227.61 | 225.60 | 226.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 226.50 | 226.24 | 226.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 228.60 | 226.71 | 226.74 | SL hit (close>static) qty=1.00 sl=228.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 228.30 | 227.03 | 226.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 232.20 | 228.25 | 227.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 238.90 | 240.23 | 237.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:00:00 | 238.90 | 240.23 | 237.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 236.98 | 239.51 | 237.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 236.98 | 239.51 | 237.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 238.35 | 239.28 | 237.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 239.80 | 239.28 | 237.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:45:00 | 238.79 | 239.02 | 237.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 239.05 | 239.13 | 237.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 11:15:00 | 263.78 | 257.14 | 251.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 252.67 | 255.68 | 255.77 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 257.36 | 255.94 | 255.85 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 254.05 | 256.03 | 256.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 252.05 | 254.76 | 255.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 250.36 | 249.71 | 251.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 251.55 | 250.08 | 251.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 251.55 | 250.08 | 251.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 251.55 | 250.08 | 251.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 250.70 | 250.20 | 251.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 250.00 | 250.71 | 251.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 251.80 | 249.15 | 249.47 | SL hit (close>static) qty=1.00 sl=251.70 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 10:15:00 | 252.35 | 249.79 | 249.73 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 248.25 | 249.45 | 249.61 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 251.96 | 249.95 | 249.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 254.04 | 252.19 | 251.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 256.00 | 256.42 | 254.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 256.90 | 256.42 | 254.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 255.16 | 256.16 | 254.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 255.16 | 256.16 | 254.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 261.60 | 263.28 | 262.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 261.60 | 263.28 | 262.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 262.15 | 263.06 | 262.38 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 261.05 | 261.93 | 261.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 258.65 | 260.99 | 261.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 13:15:00 | 260.15 | 260.00 | 260.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 13:15:00 | 260.15 | 260.00 | 260.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 260.15 | 260.00 | 260.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 261.10 | 260.00 | 260.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 261.15 | 260.18 | 260.66 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 262.40 | 261.06 | 260.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 262.55 | 261.39 | 261.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 14:15:00 | 261.55 | 261.90 | 261.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 261.55 | 261.90 | 261.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 261.55 | 261.90 | 261.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 261.55 | 261.90 | 261.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 261.20 | 261.76 | 261.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 258.60 | 261.76 | 261.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 259.00 | 261.21 | 261.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 257.45 | 260.46 | 260.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 259.20 | 258.99 | 259.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 263.25 | 258.99 | 259.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 262.80 | 259.76 | 260.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 263.45 | 259.76 | 260.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 263.10 | 260.42 | 260.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 263.55 | 261.66 | 261.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 264.35 | 264.44 | 263.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 264.35 | 264.44 | 263.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 263.10 | 264.17 | 263.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 263.10 | 264.17 | 263.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 262.95 | 263.93 | 263.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 263.05 | 263.93 | 263.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 262.95 | 263.73 | 263.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 262.95 | 263.73 | 263.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 262.25 | 263.44 | 263.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 261.60 | 263.44 | 263.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 263.65 | 263.34 | 263.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 264.00 | 263.44 | 263.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 265.40 | 263.94 | 263.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 262.45 | 265.88 | 266.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 262.45 | 265.88 | 266.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 260.10 | 262.82 | 264.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 260.70 | 258.75 | 260.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 260.70 | 258.75 | 260.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 260.70 | 258.75 | 260.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 260.70 | 258.75 | 260.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 264.15 | 259.83 | 260.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 264.15 | 259.83 | 260.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 265.25 | 260.91 | 261.01 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 264.15 | 261.56 | 261.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 267.50 | 263.36 | 262.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 308.80 | 309.40 | 301.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 308.80 | 309.40 | 301.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 307.65 | 309.53 | 307.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:30:00 | 307.40 | 309.53 | 307.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 307.05 | 308.74 | 307.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 302.80 | 308.74 | 307.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 302.75 | 307.54 | 306.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 300.75 | 307.54 | 306.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 303.00 | 305.83 | 306.03 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 309.35 | 305.76 | 305.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 310.45 | 306.70 | 306.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 306.45 | 307.04 | 306.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 306.45 | 307.04 | 306.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 306.45 | 307.04 | 306.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 306.45 | 307.04 | 306.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 305.80 | 306.79 | 306.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 305.80 | 306.79 | 306.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 304.30 | 306.30 | 306.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 304.30 | 306.30 | 306.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 305.05 | 305.93 | 305.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 304.60 | 305.66 | 305.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 305.20 | 304.79 | 305.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 305.20 | 304.79 | 305.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 305.20 | 304.79 | 305.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 304.95 | 304.79 | 305.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 305.60 | 304.95 | 305.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 305.40 | 304.95 | 305.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 305.25 | 305.01 | 305.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:15:00 | 305.55 | 305.01 | 305.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 305.55 | 305.12 | 305.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 304.10 | 305.12 | 305.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 302.60 | 304.62 | 305.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 301.90 | 303.57 | 304.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 299.40 | 300.11 | 300.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 302.45 | 301.08 | 300.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 302.45 | 301.08 | 300.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 304.90 | 302.17 | 301.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 305.50 | 305.54 | 303.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:00:00 | 305.50 | 305.54 | 303.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 314.10 | 315.58 | 314.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 313.40 | 315.58 | 314.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 313.40 | 315.14 | 313.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 313.05 | 315.14 | 313.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 315.70 | 315.25 | 314.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:30:00 | 317.75 | 315.91 | 314.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 319.35 | 321.36 | 321.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 319.35 | 321.36 | 321.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 318.75 | 320.84 | 321.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 321.85 | 320.77 | 321.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 321.85 | 320.77 | 321.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 321.85 | 320.77 | 321.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 322.60 | 320.77 | 321.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 322.45 | 321.11 | 321.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 321.85 | 321.11 | 321.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 321.35 | 321.15 | 321.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 322.15 | 321.15 | 321.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 321.10 | 321.14 | 321.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 321.10 | 321.14 | 321.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 318.95 | 320.70 | 321.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:00:00 | 318.70 | 320.22 | 320.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 318.50 | 319.88 | 320.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 318.60 | 319.65 | 320.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 318.55 | 319.45 | 320.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 319.30 | 319.42 | 320.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 319.30 | 319.42 | 320.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 320.75 | 319.68 | 320.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 320.45 | 319.68 | 320.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 321.75 | 320.10 | 320.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 321.75 | 320.10 | 320.30 | SL hit (close>static) qty=1.00 sl=321.60 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 320.65 | 316.57 | 316.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 321.35 | 318.77 | 317.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 324.50 | 324.52 | 322.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 324.50 | 324.36 | 323.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 324.50 | 324.36 | 323.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 323.70 | 324.36 | 323.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 325.50 | 329.10 | 328.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 327.00 | 329.10 | 328.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 325.10 | 328.30 | 328.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 325.10 | 328.30 | 328.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 324.30 | 327.50 | 327.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 323.50 | 325.04 | 326.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 329.00 | 325.60 | 326.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 329.00 | 325.60 | 326.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 329.00 | 325.60 | 326.04 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 327.20 | 326.40 | 326.35 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 325.45 | 326.59 | 326.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 323.25 | 325.92 | 326.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 323.60 | 323.53 | 324.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 323.00 | 323.53 | 324.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 323.50 | 323.52 | 324.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:15:00 | 325.55 | 323.52 | 324.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 325.25 | 323.87 | 324.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 325.10 | 323.87 | 324.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 324.70 | 324.03 | 324.44 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 326.45 | 324.88 | 324.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 327.45 | 325.39 | 325.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 326.05 | 326.10 | 325.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 326.05 | 326.10 | 325.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 338.05 | 339.96 | 337.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 338.55 | 339.96 | 337.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 338.20 | 339.60 | 337.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 337.15 | 339.60 | 337.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 337.25 | 339.13 | 337.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 337.25 | 339.13 | 337.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 338.15 | 338.94 | 337.84 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 336.30 | 337.36 | 337.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 331.50 | 335.79 | 336.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 334.75 | 334.69 | 335.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 14:00:00 | 334.75 | 334.69 | 335.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 326.15 | 325.70 | 326.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 326.95 | 325.70 | 326.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 325.60 | 325.68 | 326.60 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 329.45 | 327.20 | 327.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 330.70 | 327.90 | 327.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 11:15:00 | 328.20 | 328.69 | 327.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 12:00:00 | 328.20 | 328.69 | 327.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 327.75 | 328.50 | 327.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 327.75 | 328.50 | 327.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 326.90 | 328.18 | 327.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 327.25 | 328.18 | 327.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 328.60 | 328.27 | 327.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 330.95 | 329.13 | 328.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-16 14:15:00 | 364.05 | 351.60 | 350.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 340.30 | 349.34 | 350.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 336.85 | 341.27 | 344.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 328.20 | 327.65 | 331.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:00:00 | 328.20 | 327.65 | 331.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 331.95 | 328.74 | 330.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 331.95 | 328.74 | 330.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 334.10 | 329.81 | 331.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 334.35 | 329.81 | 331.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 334.25 | 331.76 | 331.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 12:15:00 | 335.00 | 333.18 | 332.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 331.25 | 333.41 | 332.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 331.25 | 333.41 | 332.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 331.25 | 333.41 | 332.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 331.25 | 333.41 | 332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 329.85 | 332.70 | 332.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 329.85 | 332.70 | 332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 330.40 | 332.24 | 332.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 327.55 | 330.48 | 331.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 12:15:00 | 331.10 | 329.95 | 330.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 12:15:00 | 331.10 | 329.95 | 330.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 331.10 | 329.95 | 330.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 331.10 | 329.95 | 330.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 330.25 | 330.01 | 330.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 328.00 | 329.71 | 330.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 311.60 | 314.83 | 318.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 307.90 | 307.20 | 310.88 | SL hit (close>ema200) qty=0.50 sl=307.20 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 310.40 | 306.75 | 306.38 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 304.55 | 306.91 | 306.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 302.95 | 306.12 | 306.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 301.90 | 301.70 | 303.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:45:00 | 302.10 | 301.70 | 303.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 303.25 | 301.54 | 302.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 303.25 | 301.54 | 302.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 304.15 | 302.06 | 303.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 301.50 | 302.06 | 303.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 306.15 | 303.03 | 303.36 | SL hit (close>static) qty=1.00 sl=305.25 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 306.65 | 303.75 | 303.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 307.65 | 304.53 | 304.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 13:15:00 | 307.35 | 308.22 | 306.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:30:00 | 307.40 | 308.22 | 306.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 306.60 | 307.90 | 306.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:15:00 | 304.50 | 307.90 | 306.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 304.50 | 307.22 | 306.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 305.15 | 307.22 | 306.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 306.95 | 307.16 | 306.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 307.25 | 307.13 | 306.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 307.75 | 307.13 | 306.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:00:00 | 307.20 | 307.18 | 306.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 307.20 | 306.85 | 306.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 306.85 | 307.26 | 306.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 307.20 | 307.26 | 306.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 307.05 | 307.22 | 306.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 305.25 | 307.22 | 306.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 303.25 | 306.42 | 306.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 303.25 | 306.42 | 306.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 301.85 | 304.33 | 305.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 303.30 | 302.92 | 304.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 303.30 | 302.92 | 304.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 302.80 | 302.90 | 304.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 302.25 | 302.90 | 304.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 304.55 | 302.63 | 303.61 | SL hit (close>static) qty=1.00 sl=304.25 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 306.05 | 303.88 | 303.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 306.40 | 304.76 | 304.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 303.55 | 305.13 | 304.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 303.55 | 305.13 | 304.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 303.55 | 305.13 | 304.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 302.80 | 305.13 | 304.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 303.60 | 304.82 | 304.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:30:00 | 303.35 | 304.82 | 304.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 301.00 | 303.69 | 303.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 300.95 | 302.65 | 303.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 303.65 | 302.16 | 302.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 303.65 | 302.16 | 302.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 303.65 | 302.16 | 302.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 303.00 | 302.16 | 302.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 302.05 | 302.14 | 302.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 301.70 | 302.00 | 302.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 301.95 | 301.94 | 302.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 301.10 | 301.94 | 302.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 286.61 | 292.06 | 294.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 286.85 | 292.06 | 294.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 286.05 | 292.06 | 294.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 287.80 | 287.51 | 290.21 | SL hit (close>ema200) qty=0.50 sl=287.51 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 290.60 | 288.91 | 288.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 292.65 | 289.83 | 289.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 287.75 | 295.72 | 294.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 287.75 | 295.72 | 294.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 287.75 | 295.72 | 294.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 287.75 | 295.72 | 294.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 283.80 | 293.34 | 293.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 283.30 | 284.81 | 287.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 285.00 | 284.71 | 286.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 285.00 | 284.71 | 286.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 286.25 | 285.02 | 286.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 285.80 | 285.02 | 286.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 285.65 | 286.26 | 286.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:45:00 | 285.75 | 286.20 | 286.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 287.05 | 286.48 | 286.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 287.05 | 286.48 | 286.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 11:15:00 | 285.40 | 286.38 | 286.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 284.15 | 285.67 | 286.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 285.45 | 285.40 | 285.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 285.45 | 285.40 | 285.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 285.45 | 285.40 | 285.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 282.70 | 284.88 | 285.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 283.00 | 284.53 | 285.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 283.00 | 284.53 | 285.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 282.95 | 284.17 | 284.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 285.05 | 283.39 | 284.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 285.15 | 283.39 | 284.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 285.40 | 283.79 | 284.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 285.40 | 283.79 | 284.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 281.90 | 283.44 | 283.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 277.40 | 283.18 | 283.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 284.30 | 280.01 | 279.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 284.30 | 280.01 | 279.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 287.50 | 283.56 | 281.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 284.30 | 284.56 | 283.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:30:00 | 284.05 | 284.56 | 283.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 281.00 | 283.74 | 283.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 279.80 | 283.74 | 283.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 281.35 | 283.26 | 282.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 280.80 | 283.26 | 282.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 282.15 | 282.65 | 282.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 279.10 | 281.75 | 282.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 280.15 | 279.80 | 280.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 280.15 | 279.80 | 280.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 280.15 | 279.80 | 280.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 280.30 | 279.80 | 280.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 284.05 | 280.65 | 281.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 284.05 | 280.65 | 281.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 283.80 | 281.28 | 281.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 284.25 | 281.28 | 281.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 280.50 | 280.90 | 281.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 281.30 | 280.90 | 281.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 285.00 | 281.72 | 281.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 09:15:00 | 291.05 | 284.55 | 283.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 12:15:00 | 285.35 | 285.66 | 284.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:00:00 | 285.35 | 285.66 | 284.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 285.25 | 285.48 | 284.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 285.25 | 285.48 | 284.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 284.70 | 285.33 | 284.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 280.00 | 285.33 | 284.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 282.80 | 284.82 | 284.14 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 281.60 | 283.40 | 283.56 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 285.30 | 283.90 | 283.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 293.20 | 285.76 | 284.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 292.60 | 295.58 | 292.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 292.60 | 295.58 | 292.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 292.60 | 295.58 | 292.64 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 288.00 | 291.48 | 291.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 285.30 | 290.25 | 291.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 10:15:00 | 274.45 | 273.57 | 278.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 11:00:00 | 274.45 | 273.57 | 278.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 284.45 | 275.91 | 278.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 285.55 | 275.91 | 278.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 282.50 | 277.22 | 278.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 284.45 | 277.22 | 278.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 284.00 | 280.24 | 279.95 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 276.35 | 279.34 | 279.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 275.85 | 278.64 | 279.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 256.70 | 256.32 | 262.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 256.70 | 256.32 | 262.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 263.40 | 258.15 | 261.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 263.40 | 258.15 | 261.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 263.35 | 259.19 | 262.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 263.35 | 259.19 | 262.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 263.70 | 260.09 | 262.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:00:00 | 263.70 | 260.09 | 262.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 265.70 | 261.21 | 262.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 266.10 | 261.21 | 262.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 268.35 | 263.52 | 263.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 269.75 | 265.51 | 264.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 272.60 | 273.05 | 270.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 272.60 | 273.05 | 270.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 272.60 | 273.05 | 270.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 275.40 | 272.75 | 271.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 269.35 | 271.40 | 270.76 | SL hit (close<static) qty=1.00 sl=269.55 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 283.55 | 284.48 | 284.50 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 286.60 | 284.90 | 284.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 295.80 | 288.82 | 286.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 301.80 | 301.90 | 297.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 301.80 | 301.90 | 297.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 295.50 | 300.35 | 297.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 298.85 | 298.64 | 297.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 298.60 | 298.64 | 297.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 288.70 | 296.45 | 296.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 288.70 | 296.45 | 296.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 287.20 | 294.60 | 296.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 15:15:00 | 281.80 | 281.77 | 285.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 281.55 | 281.77 | 285.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 278.45 | 281.10 | 284.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 277.90 | 281.10 | 284.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 264.00 | 266.07 | 269.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-25 13:15:00 | 250.11 | 253.31 | 258.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 229.57 | 220.72 | 220.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 10:15:00 | 230.92 | 222.76 | 221.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 231.85 | 238.24 | 233.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 231.85 | 238.24 | 233.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 231.85 | 238.24 | 233.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 231.85 | 238.24 | 233.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 230.13 | 236.62 | 233.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 230.13 | 236.62 | 233.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 229.50 | 231.69 | 231.86 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 234.67 | 232.43 | 232.17 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 228.05 | 231.63 | 231.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 226.04 | 229.95 | 231.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 231.77 | 228.53 | 229.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 231.77 | 228.53 | 229.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 231.77 | 228.53 | 229.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 233.49 | 228.53 | 229.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 235.29 | 230.45 | 230.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 237.48 | 231.85 | 231.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 235.22 | 239.47 | 237.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 235.22 | 239.47 | 237.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 235.22 | 239.47 | 237.10 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 232.71 | 236.13 | 236.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 229.56 | 234.82 | 235.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 238.83 | 232.41 | 233.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 238.83 | 232.41 | 233.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 238.83 | 232.41 | 233.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 238.83 | 232.41 | 233.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 234.74 | 232.87 | 233.63 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 239.08 | 234.86 | 234.45 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 226.30 | 233.88 | 234.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 225.12 | 232.13 | 233.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 231.65 | 229.80 | 231.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 14:15:00 | 231.65 | 229.80 | 231.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 231.65 | 229.80 | 231.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 231.65 | 229.80 | 231.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 231.99 | 230.24 | 231.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 229.78 | 230.24 | 231.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 234.50 | 231.09 | 231.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 234.50 | 231.09 | 231.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 231.54 | 231.18 | 231.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 228.58 | 232.14 | 232.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 10:45:00 | 229.98 | 231.52 | 231.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 230.30 | 231.33 | 231.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 243.72 | 233.98 | 232.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 243.72 | 233.98 | 232.84 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 236.85 | 238.82 | 238.89 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 246.20 | 239.54 | 239.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 247.17 | 243.17 | 241.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 251.04 | 251.45 | 249.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 251.04 | 251.45 | 249.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 251.04 | 251.45 | 249.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 254.33 | 251.94 | 249.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 252.74 | 258.08 | 258.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 252.74 | 258.08 | 258.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 247.59 | 254.37 | 255.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 259.00 | 251.12 | 253.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 15:15:00 | 259.00 | 251.12 | 253.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 259.00 | 251.12 | 253.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 257.78 | 251.12 | 253.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 257.19 | 252.34 | 253.38 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 258.17 | 254.15 | 254.06 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 244.28 | 252.55 | 253.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 242.90 | 250.62 | 252.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 249.00 | 247.90 | 249.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:00:00 | 249.00 | 247.90 | 249.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 251.00 | 248.78 | 250.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 251.85 | 248.78 | 250.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 250.04 | 249.03 | 250.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:30:00 | 251.26 | 249.03 | 250.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 250.44 | 249.31 | 250.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 250.44 | 249.31 | 250.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 252.10 | 249.87 | 250.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 252.10 | 249.87 | 250.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 247.85 | 249.84 | 250.18 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 251.76 | 249.98 | 249.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 256.20 | 252.06 | 250.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 255.24 | 256.16 | 254.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 255.24 | 256.16 | 254.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 255.24 | 256.16 | 254.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 255.30 | 256.16 | 254.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 254.13 | 255.60 | 254.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 253.98 | 255.60 | 254.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 254.56 | 255.40 | 254.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 254.25 | 255.40 | 254.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 255.64 | 255.44 | 254.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 256.48 | 255.62 | 254.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 13:15:00 | 242.48 | 2025-05-19 13:15:00 | 240.13 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-29 12:30:00 | 226.50 | 2025-05-29 13:15:00 | 228.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-06-04 09:15:00 | 239.80 | 2025-06-06 11:15:00 | 263.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:45:00 | 238.79 | 2025-06-06 11:15:00 | 262.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 10:30:00 | 239.05 | 2025-06-06 11:15:00 | 262.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:45:00 | 250.00 | 2025-06-19 09:15:00 | 251.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-11 12:30:00 | 264.00 | 2025-07-16 10:15:00 | 262.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-14 09:30:00 | 265.40 | 2025-07-16 10:15:00 | 262.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-05 14:45:00 | 301.90 | 2025-08-11 09:15:00 | 302.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-08-08 09:15:00 | 299.40 | 2025-08-11 09:15:00 | 302.45 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-19 11:30:00 | 317.75 | 2025-08-22 14:15:00 | 319.35 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-08-26 10:00:00 | 318.70 | 2025-08-28 09:15:00 | 321.75 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-08-26 11:00:00 | 318.50 | 2025-08-28 09:15:00 | 321.75 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-26 11:30:00 | 318.60 | 2025-08-28 09:15:00 | 321.75 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-26 14:15:00 | 318.55 | 2025-08-28 09:15:00 | 321.75 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-28 12:00:00 | 319.95 | 2025-09-01 12:15:00 | 320.65 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-10-06 09:30:00 | 330.95 | 2025-10-16 14:15:00 | 364.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 328.00 | 2025-11-06 09:15:00 | 311.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 328.00 | 2025-11-07 12:15:00 | 307.90 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2025-11-17 09:15:00 | 301.50 | 2025-11-17 10:15:00 | 306.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-19 10:30:00 | 307.25 | 2025-11-21 09:15:00 | 303.25 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-19 11:15:00 | 307.75 | 2025-11-21 09:15:00 | 303.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-19 13:00:00 | 307.20 | 2025-11-21 09:15:00 | 303.25 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-20 11:15:00 | 307.20 | 2025-11-21 09:15:00 | 303.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-24 14:15:00 | 302.25 | 2025-11-25 09:15:00 | 304.55 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-25 10:45:00 | 302.30 | 2025-11-26 09:15:00 | 305.25 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-25 11:45:00 | 302.40 | 2025-11-26 09:15:00 | 305.25 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-25 14:45:00 | 302.05 | 2025-11-26 09:15:00 | 305.25 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-01 11:30:00 | 301.70 | 2025-12-08 10:15:00 | 286.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:15:00 | 301.95 | 2025-12-08 10:15:00 | 286.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 301.10 | 2025-12-08 10:15:00 | 286.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:30:00 | 301.70 | 2025-12-09 11:15:00 | 287.80 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-12-01 14:15:00 | 301.95 | 2025-12-09 11:15:00 | 287.80 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-12-02 09:15:00 | 301.10 | 2025-12-09 11:15:00 | 287.80 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-12-19 10:15:00 | 285.80 | 2025-12-22 15:15:00 | 287.05 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-12-22 10:15:00 | 285.65 | 2025-12-22 15:15:00 | 287.05 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-12-22 12:45:00 | 285.75 | 2025-12-22 15:15:00 | 287.05 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-26 09:15:00 | 282.70 | 2026-01-01 10:15:00 | 284.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-26 09:45:00 | 283.00 | 2026-01-01 10:15:00 | 284.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-26 10:15:00 | 283.00 | 2026-01-01 10:15:00 | 284.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-26 12:15:00 | 282.95 | 2026-01-01 10:15:00 | 284.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-30 09:15:00 | 277.40 | 2026-01-01 10:15:00 | 284.30 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-01 11:45:00 | 275.40 | 2026-02-01 14:15:00 | 269.35 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-02-03 09:15:00 | 287.15 | 2026-02-06 15:15:00 | 283.55 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-12 12:30:00 | 298.85 | 2026-02-13 09:15:00 | 288.70 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-02-12 13:15:00 | 298.60 | 2026-02-13 09:15:00 | 288.70 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2026-02-18 10:15:00 | 277.90 | 2026-02-24 09:15:00 | 264.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 277.90 | 2026-02-25 13:15:00 | 250.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-07 09:15:00 | 228.58 | 2026-04-08 09:15:00 | 243.72 | STOP_HIT | 1.00 | -6.62% |
| SELL | retest2 | 2026-04-07 10:45:00 | 229.98 | 2026-04-08 09:15:00 | 243.72 | STOP_HIT | 1.00 | -5.97% |
| SELL | retest2 | 2026-04-07 13:45:00 | 230.30 | 2026-04-08 09:15:00 | 243.72 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2026-04-20 10:30:00 | 254.33 | 2026-04-24 10:15:00 | 252.74 | STOP_HIT | 1.00 | -0.63% |
