# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 456.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 72 |
| ALERT3 | 476 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 194 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 193 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 208 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 148
- **Target hits / Stop hits / Partials:** 5 / 193 / 10
- **Avg / median % per leg:** 0.29% / -0.59%
- **Sum % (uncompounded):** 61.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 99 | 30 | 30.3% | 5 | 94 | 0 | 0.36% | 35.8% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | 0.15% | 0.6% |
| BUY @ 3rd Alert (retest2) | 95 | 29 | 30.5% | 5 | 90 | 0 | 0.37% | 35.2% |
| SELL (all) | 109 | 30 | 27.5% | 0 | 99 | 10 | 0.23% | 25.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 109 | 30 | 27.5% | 0 | 99 | 10 | 0.23% | 25.3% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | 0.15% | 0.6% |
| retest2 (combined) | 204 | 59 | 28.9% | 5 | 189 | 10 | 0.30% | 60.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 237.70 | 236.26 | 236.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 11:15:00 | 238.00 | 236.61 | 236.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 15:15:00 | 241.50 | 241.53 | 240.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 09:15:00 | 241.10 | 241.53 | 240.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 239.10 | 241.26 | 240.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 15:00:00 | 239.10 | 241.26 | 240.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 15:15:00 | 240.00 | 241.01 | 240.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 09:15:00 | 238.35 | 241.01 | 240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 237.40 | 240.29 | 240.39 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 12:15:00 | 240.65 | 240.43 | 240.43 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 13:15:00 | 239.70 | 240.28 | 240.36 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 241.00 | 240.38 | 240.35 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 13:15:00 | 239.25 | 240.13 | 240.24 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 12:15:00 | 240.35 | 239.80 | 239.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 13:15:00 | 241.20 | 240.08 | 239.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 10:15:00 | 239.65 | 240.23 | 240.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 10:15:00 | 239.65 | 240.23 | 240.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 239.65 | 240.23 | 240.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:45:00 | 239.75 | 240.23 | 240.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 239.25 | 240.04 | 239.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:00:00 | 239.25 | 240.04 | 239.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 12:15:00 | 238.05 | 239.64 | 239.81 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 242.15 | 240.30 | 240.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 242.30 | 241.29 | 240.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 11:15:00 | 244.10 | 245.05 | 243.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 12:00:00 | 244.10 | 245.05 | 243.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 244.40 | 244.92 | 243.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:45:00 | 243.85 | 244.92 | 243.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 243.75 | 244.69 | 243.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 14:00:00 | 243.75 | 244.69 | 243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 14:15:00 | 244.30 | 244.61 | 243.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 14:30:00 | 244.45 | 244.61 | 243.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 244.45 | 244.58 | 243.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:15:00 | 243.15 | 244.58 | 243.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 240.45 | 243.75 | 243.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:00:00 | 240.45 | 243.75 | 243.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 240.65 | 243.13 | 243.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 09:15:00 | 230.50 | 239.60 | 241.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 11:15:00 | 232.00 | 231.92 | 235.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-02 12:00:00 | 232.00 | 231.92 | 235.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 230.00 | 228.73 | 230.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:00:00 | 230.00 | 228.73 | 230.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 230.10 | 229.01 | 230.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 11:00:00 | 230.10 | 229.01 | 230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 230.25 | 229.26 | 230.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:00:00 | 230.25 | 229.26 | 230.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 230.75 | 229.55 | 230.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 13:00:00 | 230.75 | 229.55 | 230.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 13:15:00 | 230.45 | 229.73 | 230.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 14:15:00 | 231.15 | 229.73 | 230.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 230.65 | 229.92 | 230.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 14:30:00 | 231.40 | 229.92 | 230.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 230.00 | 230.11 | 230.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 10:30:00 | 230.30 | 230.11 | 230.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 228.50 | 228.36 | 228.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 11:30:00 | 228.20 | 228.65 | 228.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 12:30:00 | 228.25 | 228.70 | 228.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 15:15:00 | 229.05 | 228.82 | 228.83 | SL hit (close>static) qty=1.00 sl=228.95 alert=retest2 |

### Cycle 11 — BUY (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 15:15:00 | 229.00 | 228.82 | 228.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 229.30 | 228.92 | 228.85 | Break + close above crossover candle high |

### Cycle 12 — SELL (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 10:15:00 | 228.20 | 228.77 | 228.79 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 229.00 | 228.75 | 228.72 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 09:15:00 | 228.30 | 228.72 | 228.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 10:15:00 | 227.45 | 228.46 | 228.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 227.25 | 226.99 | 227.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 15:00:00 | 227.25 | 226.99 | 227.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 227.40 | 227.07 | 227.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 227.10 | 227.07 | 227.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 228.10 | 227.28 | 227.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:45:00 | 227.85 | 227.28 | 227.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 227.60 | 227.34 | 227.55 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 14:15:00 | 228.25 | 227.72 | 227.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 15:15:00 | 228.50 | 227.88 | 227.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 228.00 | 228.30 | 228.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 228.00 | 228.30 | 228.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 228.00 | 228.30 | 228.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 228.00 | 228.30 | 228.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 227.75 | 228.19 | 227.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 227.75 | 228.19 | 227.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 227.15 | 227.98 | 227.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:00:00 | 227.15 | 227.98 | 227.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 226.95 | 227.77 | 227.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 226.00 | 227.31 | 227.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 224.70 | 224.64 | 225.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 224.70 | 224.64 | 225.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 224.70 | 224.64 | 225.49 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 12:15:00 | 226.20 | 225.55 | 225.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 13:15:00 | 226.60 | 225.76 | 225.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 231.15 | 231.19 | 229.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 11:15:00 | 232.35 | 231.33 | 230.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 231.95 | 231.63 | 230.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:00:00 | 232.15 | 231.72 | 230.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 230.90 | 231.56 | 230.91 | SL hit (close<ema400) qty=1.00 sl=230.91 alert=retest1 |

### Cycle 18 — SELL (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 09:15:00 | 232.20 | 234.10 | 234.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 10:15:00 | 231.15 | 233.51 | 233.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 10:15:00 | 231.40 | 231.30 | 232.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 10:30:00 | 231.60 | 231.30 | 232.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 230.85 | 230.93 | 231.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 230.85 | 230.93 | 231.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 231.95 | 231.14 | 231.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 231.85 | 231.14 | 231.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 231.05 | 231.12 | 231.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 232.35 | 231.12 | 231.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 231.70 | 231.23 | 231.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:45:00 | 231.45 | 231.23 | 231.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 231.45 | 231.28 | 231.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:30:00 | 231.55 | 231.28 | 231.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 230.85 | 231.19 | 231.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 13:45:00 | 230.30 | 231.07 | 231.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 14:45:00 | 230.00 | 230.95 | 231.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 10:15:00 | 230.50 | 230.84 | 231.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:00:00 | 230.50 | 230.77 | 231.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 228.80 | 229.21 | 229.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:45:00 | 229.85 | 229.21 | 229.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 229.15 | 228.99 | 229.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:45:00 | 229.40 | 228.99 | 229.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 229.65 | 229.12 | 229.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:45:00 | 229.85 | 229.12 | 229.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 230.10 | 229.32 | 229.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:00:00 | 230.10 | 229.32 | 229.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 230.10 | 229.47 | 229.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 230.10 | 229.47 | 229.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 229.65 | 229.55 | 229.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 230.00 | 229.55 | 229.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 229.75 | 229.59 | 229.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 14:30:00 | 228.60 | 229.17 | 229.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 10:15:00 | 231.15 | 229.54 | 229.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 231.15 | 229.54 | 229.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 11:15:00 | 231.50 | 229.93 | 229.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 230.75 | 230.92 | 230.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 13:00:00 | 230.75 | 230.92 | 230.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 229.15 | 230.56 | 230.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:00:00 | 229.15 | 230.56 | 230.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 229.95 | 230.44 | 230.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 230.75 | 230.34 | 230.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 12:15:00 | 230.15 | 230.28 | 230.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 12:15:00 | 229.50 | 230.12 | 230.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 12:15:00 | 229.50 | 230.12 | 230.20 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 15:15:00 | 230.75 | 230.27 | 230.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 09:15:00 | 231.20 | 230.45 | 230.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 10:15:00 | 229.75 | 230.31 | 230.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 10:15:00 | 229.75 | 230.31 | 230.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 229.75 | 230.31 | 230.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:00:00 | 229.75 | 230.31 | 230.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 229.85 | 230.22 | 230.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 229.20 | 229.89 | 230.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 229.10 | 228.38 | 229.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 229.10 | 228.38 | 229.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 229.10 | 228.38 | 229.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 229.10 | 228.38 | 229.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 228.95 | 228.49 | 228.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:15:00 | 229.55 | 228.49 | 228.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 229.00 | 228.59 | 228.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 12:15:00 | 228.70 | 228.59 | 228.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 09:15:00 | 231.25 | 229.18 | 229.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 231.25 | 229.18 | 229.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 10:15:00 | 236.00 | 230.55 | 229.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 235.65 | 237.10 | 234.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:00:00 | 235.65 | 237.10 | 234.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 234.30 | 236.32 | 234.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 234.30 | 236.32 | 234.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 232.50 | 235.56 | 234.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 232.50 | 235.56 | 234.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 234.55 | 235.36 | 234.26 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 230.45 | 233.44 | 233.69 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 14:15:00 | 233.65 | 233.17 | 233.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 15:15:00 | 234.40 | 233.42 | 233.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 10:15:00 | 232.85 | 233.40 | 233.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 10:15:00 | 232.85 | 233.40 | 233.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 232.85 | 233.40 | 233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 232.85 | 233.40 | 233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 232.80 | 233.28 | 233.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 12:00:00 | 232.80 | 233.28 | 233.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 233.50 | 233.35 | 233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:15:00 | 233.40 | 233.35 | 233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 232.95 | 233.27 | 233.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:45:00 | 232.75 | 233.27 | 233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 15:15:00 | 233.00 | 233.22 | 233.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 09:15:00 | 232.20 | 233.01 | 233.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 232.20 | 231.11 | 231.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 232.20 | 231.11 | 231.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 232.20 | 231.11 | 231.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:45:00 | 234.00 | 231.11 | 231.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 231.20 | 231.13 | 231.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 232.10 | 231.13 | 231.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 232.85 | 231.47 | 231.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 12:00:00 | 232.85 | 231.47 | 231.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 233.90 | 231.96 | 232.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 12:45:00 | 234.00 | 231.96 | 232.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 13:15:00 | 233.00 | 232.17 | 232.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 236.05 | 233.67 | 232.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 234.05 | 234.64 | 233.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 234.05 | 234.64 | 233.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 234.05 | 234.64 | 233.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:45:00 | 234.00 | 234.64 | 233.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 234.05 | 234.52 | 233.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:00:00 | 234.05 | 234.52 | 233.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 234.05 | 234.43 | 233.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:45:00 | 233.95 | 234.43 | 233.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 234.65 | 234.47 | 234.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 14:30:00 | 235.05 | 234.48 | 234.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 15:15:00 | 235.30 | 234.48 | 234.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 09:15:00 | 232.20 | 234.16 | 234.03 | SL hit (close<static) qty=1.00 sl=233.90 alert=retest2 |

### Cycle 28 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 233.35 | 233.83 | 233.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 10:15:00 | 232.65 | 233.62 | 233.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 15:15:00 | 233.25 | 233.24 | 233.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 09:15:00 | 232.30 | 233.24 | 233.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 232.05 | 233.01 | 233.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 11:45:00 | 231.35 | 232.58 | 233.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 09:15:00 | 228.70 | 232.37 | 232.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 12:15:00 | 231.00 | 230.38 | 230.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 231.00 | 230.38 | 230.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 232.10 | 231.13 | 230.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 230.50 | 231.01 | 230.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 10:15:00 | 230.50 | 231.01 | 230.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 230.50 | 231.01 | 230.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:00:00 | 230.50 | 231.01 | 230.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 230.30 | 230.87 | 230.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:45:00 | 230.10 | 230.87 | 230.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 229.40 | 230.57 | 230.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 229.15 | 230.25 | 230.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 229.35 | 228.62 | 229.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 229.35 | 228.62 | 229.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 229.35 | 228.62 | 229.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:00:00 | 229.35 | 228.62 | 229.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 229.35 | 228.77 | 229.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:30:00 | 229.45 | 228.77 | 229.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 229.45 | 228.90 | 229.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:00:00 | 229.45 | 228.90 | 229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 229.35 | 228.99 | 229.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:45:00 | 229.50 | 228.99 | 229.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 229.50 | 229.10 | 229.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 13:45:00 | 229.40 | 229.10 | 229.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 229.45 | 229.17 | 229.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:45:00 | 229.75 | 229.17 | 229.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 229.40 | 229.21 | 229.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:15:00 | 229.65 | 229.21 | 229.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 230.15 | 229.40 | 229.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 230.00 | 229.40 | 229.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 229.90 | 229.50 | 229.46 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 14:15:00 | 228.80 | 229.46 | 229.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 15:15:00 | 228.65 | 229.30 | 229.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 229.85 | 229.41 | 229.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 229.85 | 229.41 | 229.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 229.85 | 229.41 | 229.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 229.85 | 229.41 | 229.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 230.10 | 229.55 | 229.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:45:00 | 230.00 | 229.55 | 229.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 230.00 | 229.64 | 229.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 15:15:00 | 230.30 | 229.96 | 229.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 13:15:00 | 278.65 | 279.38 | 273.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 14:00:00 | 278.65 | 279.38 | 273.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 273.00 | 277.96 | 274.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 272.25 | 277.96 | 274.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 275.10 | 277.39 | 274.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 274.15 | 277.39 | 274.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 273.90 | 276.69 | 274.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 273.90 | 276.69 | 274.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 274.20 | 276.19 | 274.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:45:00 | 273.55 | 276.19 | 274.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 13:15:00 | 273.85 | 275.73 | 274.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 13:45:00 | 272.45 | 275.73 | 274.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 15:15:00 | 270.35 | 273.79 | 273.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 09:15:00 | 275.60 | 273.79 | 273.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 273.20 | 273.79 | 273.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:30:00 | 275.20 | 274.31 | 274.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 12:00:00 | 276.35 | 274.31 | 274.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 15:15:00 | 280.30 | 282.08 | 282.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 15:15:00 | 280.30 | 282.08 | 282.12 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 284.55 | 282.24 | 282.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 14:15:00 | 288.25 | 284.72 | 283.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 286.50 | 287.39 | 285.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 286.50 | 287.39 | 285.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 286.20 | 287.15 | 286.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:15:00 | 291.15 | 287.26 | 286.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 289.75 | 291.70 | 291.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 289.75 | 291.70 | 291.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 286.85 | 290.04 | 290.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 288.95 | 287.71 | 288.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 288.95 | 287.71 | 288.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 288.95 | 287.71 | 288.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 288.95 | 287.71 | 288.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 289.20 | 288.01 | 288.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 289.20 | 288.01 | 288.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 289.40 | 288.29 | 289.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 289.10 | 288.29 | 289.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 289.65 | 288.83 | 289.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:15:00 | 288.90 | 288.83 | 289.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 12:45:00 | 289.05 | 288.95 | 289.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 14:00:00 | 289.30 | 289.02 | 289.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 09:15:00 | 291.65 | 288.41 | 288.45 | SL hit (close>static) qty=1.00 sl=290.90 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 297.90 | 290.31 | 289.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 299.60 | 293.42 | 290.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 307.50 | 307.70 | 305.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 09:15:00 | 310.30 | 307.70 | 305.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 311.50 | 315.33 | 313.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:45:00 | 311.95 | 315.33 | 313.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 312.30 | 314.73 | 313.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 11:15:00 | 313.35 | 314.73 | 313.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 12:00:00 | 313.20 | 314.42 | 313.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 12:15:00 | 310.80 | 313.14 | 313.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 310.80 | 313.14 | 313.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 308.05 | 311.33 | 312.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 309.50 | 308.84 | 310.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 09:45:00 | 309.25 | 308.84 | 310.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 307.65 | 308.60 | 310.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:30:00 | 306.40 | 307.96 | 309.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 307.05 | 308.78 | 309.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 310.35 | 306.76 | 307.70 | SL hit (close>static) qty=1.00 sl=310.30 alert=retest2 |

### Cycle 39 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 315.40 | 309.33 | 308.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 12:15:00 | 315.90 | 310.64 | 309.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 12:15:00 | 313.10 | 313.23 | 311.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 13:00:00 | 313.10 | 313.23 | 311.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 312.75 | 313.66 | 312.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 10:00:00 | 312.75 | 313.66 | 312.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 312.65 | 313.46 | 312.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 10:45:00 | 312.50 | 313.46 | 312.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 311.30 | 313.03 | 312.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:00:00 | 311.30 | 313.03 | 312.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 313.15 | 313.05 | 312.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 13:45:00 | 313.45 | 313.36 | 312.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 11:15:00 | 308.60 | 312.15 | 312.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 308.60 | 312.15 | 312.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 13:15:00 | 307.30 | 310.68 | 311.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 310.80 | 309.62 | 310.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 310.80 | 309.62 | 310.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 310.80 | 309.62 | 310.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:00:00 | 310.80 | 309.62 | 310.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 310.95 | 309.89 | 310.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 12:45:00 | 309.95 | 309.87 | 310.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 11:15:00 | 312.60 | 310.85 | 310.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 312.60 | 310.85 | 310.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 316.15 | 312.52 | 311.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 13:15:00 | 315.00 | 315.69 | 314.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 14:00:00 | 315.00 | 315.69 | 314.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 313.70 | 315.29 | 314.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 15:00:00 | 313.70 | 315.29 | 314.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 314.80 | 315.19 | 314.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 316.10 | 315.19 | 314.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-13 12:15:00 | 347.71 | 336.23 | 328.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 09:15:00 | 334.90 | 344.43 | 345.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 10:15:00 | 330.10 | 341.56 | 343.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 11:15:00 | 332.55 | 332.47 | 335.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 12:00:00 | 332.55 | 332.47 | 335.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 335.30 | 333.25 | 334.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 335.55 | 333.56 | 334.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 334.95 | 333.84 | 334.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:30:00 | 334.80 | 333.84 | 334.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 336.25 | 334.32 | 334.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:45:00 | 336.10 | 334.32 | 334.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 333.00 | 334.06 | 334.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:30:00 | 335.00 | 334.06 | 334.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 334.70 | 333.74 | 334.26 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 337.25 | 334.73 | 334.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 340.90 | 336.37 | 335.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 340.80 | 341.05 | 339.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 340.80 | 341.05 | 339.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 340.80 | 341.05 | 339.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 340.80 | 341.05 | 339.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 341.05 | 341.05 | 339.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 14:45:00 | 342.35 | 340.74 | 339.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 11:15:00 | 349.80 | 351.62 | 351.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 11:15:00 | 349.80 | 351.62 | 351.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 12:15:00 | 348.75 | 351.04 | 351.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 14:15:00 | 351.75 | 351.16 | 351.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 14:15:00 | 351.75 | 351.16 | 351.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 351.75 | 351.16 | 351.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 14:30:00 | 352.25 | 351.16 | 351.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 351.90 | 351.31 | 351.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:15:00 | 353.30 | 351.31 | 351.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 09:15:00 | 354.20 | 351.89 | 351.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 10:15:00 | 356.00 | 352.71 | 352.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 351.60 | 353.08 | 352.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 12:15:00 | 351.60 | 353.08 | 352.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 351.60 | 353.08 | 352.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 351.60 | 353.08 | 352.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 350.40 | 352.54 | 352.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 347.55 | 352.54 | 352.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 350.75 | 351.93 | 352.00 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 357.20 | 352.98 | 352.47 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 350.65 | 352.60 | 352.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 14:15:00 | 347.45 | 350.61 | 351.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 349.50 | 346.92 | 348.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 349.50 | 346.92 | 348.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 349.50 | 346.92 | 348.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 349.50 | 346.92 | 348.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 350.10 | 347.55 | 348.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 349.35 | 347.55 | 348.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 348.65 | 347.77 | 348.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:30:00 | 350.45 | 347.77 | 348.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 349.30 | 348.08 | 348.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 13:00:00 | 349.30 | 348.08 | 348.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 348.70 | 348.20 | 348.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 13:30:00 | 349.05 | 348.20 | 348.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 347.00 | 347.96 | 348.55 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 353.30 | 349.04 | 348.94 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 09:15:00 | 344.60 | 348.40 | 348.76 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 352.75 | 349.05 | 348.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 11:15:00 | 353.45 | 349.93 | 349.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 357.15 | 362.24 | 357.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 357.15 | 362.24 | 357.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 357.15 | 362.24 | 357.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 357.15 | 362.24 | 357.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 352.20 | 360.23 | 357.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 352.20 | 360.23 | 357.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 351.30 | 358.44 | 356.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 349.30 | 358.44 | 356.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2023-12-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 13:15:00 | 354.70 | 355.78 | 355.80 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 356.50 | 355.93 | 355.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 363.80 | 357.51 | 356.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 14:15:00 | 365.95 | 366.06 | 363.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 14:45:00 | 366.40 | 366.06 | 363.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 364.65 | 365.65 | 364.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 365.40 | 365.65 | 364.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 365.00 | 365.52 | 364.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 14:15:00 | 365.30 | 365.52 | 364.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 15:00:00 | 365.60 | 365.54 | 364.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 14:15:00 | 384.30 | 385.73 | 385.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 14:15:00 | 384.30 | 385.73 | 385.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 381.95 | 384.81 | 385.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 383.15 | 382.53 | 383.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 383.15 | 382.53 | 383.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 383.15 | 382.53 | 383.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:15:00 | 385.60 | 382.53 | 383.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 386.05 | 383.23 | 383.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:30:00 | 386.70 | 383.23 | 383.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 385.95 | 383.78 | 384.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:00:00 | 385.95 | 383.78 | 384.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 12:15:00 | 387.30 | 384.48 | 384.41 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 379.70 | 383.91 | 384.22 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 10:15:00 | 386.25 | 383.05 | 383.04 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 11:15:00 | 380.10 | 382.77 | 383.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 386.35 | 383.08 | 382.89 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 378.80 | 382.70 | 383.01 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 10:15:00 | 384.50 | 383.25 | 383.15 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 381.40 | 382.88 | 382.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 12:15:00 | 379.65 | 382.24 | 382.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 381.75 | 378.24 | 379.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 381.75 | 378.24 | 379.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 381.75 | 378.24 | 379.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:30:00 | 383.95 | 378.24 | 379.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 381.35 | 378.86 | 379.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 382.35 | 378.86 | 379.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 381.25 | 380.20 | 380.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 383.10 | 380.78 | 380.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 393.80 | 394.69 | 389.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 10:00:00 | 393.80 | 394.69 | 389.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 390.10 | 393.77 | 389.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:45:00 | 389.85 | 393.77 | 389.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 385.70 | 392.16 | 389.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 385.70 | 392.16 | 389.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 383.65 | 390.46 | 388.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 383.65 | 390.46 | 388.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 375.45 | 385.58 | 386.81 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 388.35 | 385.37 | 385.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 396.65 | 389.54 | 387.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 402.65 | 405.71 | 400.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 402.65 | 405.71 | 400.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 406.40 | 404.77 | 401.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 12:15:00 | 407.75 | 404.77 | 401.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:45:00 | 408.20 | 405.46 | 402.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 408.20 | 405.71 | 403.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 412.30 | 405.77 | 404.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-05 13:15:00 | 448.53 | 432.21 | 421.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 440.60 | 448.44 | 449.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 432.55 | 445.26 | 447.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 442.45 | 440.29 | 444.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 442.45 | 440.29 | 444.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 442.45 | 440.29 | 444.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:15:00 | 444.80 | 440.29 | 444.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 456.55 | 443.55 | 445.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 456.55 | 443.55 | 445.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 449.55 | 444.75 | 445.55 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 13:15:00 | 452.70 | 447.04 | 446.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 09:15:00 | 458.75 | 450.94 | 448.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 12:15:00 | 481.25 | 481.35 | 476.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 12:30:00 | 481.50 | 481.35 | 476.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 460.35 | 476.97 | 475.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 460.35 | 476.97 | 475.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 15:15:00 | 459.60 | 473.49 | 473.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 445.70 | 467.93 | 471.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 439.70 | 438.54 | 446.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 11:45:00 | 438.00 | 438.54 | 446.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 448.00 | 441.46 | 445.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 446.80 | 441.46 | 445.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 447.50 | 442.67 | 445.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:45:00 | 444.45 | 442.70 | 445.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 13:15:00 | 444.20 | 443.03 | 444.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 15:00:00 | 443.70 | 443.48 | 444.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:45:00 | 443.25 | 443.93 | 444.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 446.65 | 444.48 | 444.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:00:00 | 446.65 | 444.48 | 444.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 443.05 | 444.19 | 444.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 12:30:00 | 442.50 | 444.18 | 444.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 15:00:00 | 441.95 | 443.90 | 444.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 09:45:00 | 442.45 | 443.32 | 444.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 11:30:00 | 442.50 | 442.10 | 443.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 440.25 | 440.83 | 442.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:45:00 | 440.10 | 440.83 | 442.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 435.40 | 439.74 | 441.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 12:15:00 | 431.65 | 438.69 | 440.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 10:15:00 | 432.95 | 435.38 | 438.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 11:15:00 | 432.15 | 434.98 | 437.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 443.55 | 437.58 | 437.81 | SL hit (close>static) qty=1.00 sl=441.60 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 443.85 | 438.83 | 438.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 15:15:00 | 447.65 | 442.69 | 440.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 448.35 | 456.03 | 453.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 448.35 | 456.03 | 453.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 448.35 | 456.03 | 453.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 448.35 | 456.03 | 453.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 453.55 | 455.53 | 453.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 12:30:00 | 455.85 | 455.30 | 453.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 14:15:00 | 453.10 | 457.00 | 457.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 14:15:00 | 453.10 | 457.00 | 457.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 451.35 | 454.60 | 455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 425.40 | 425.27 | 434.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:45:00 | 425.90 | 425.27 | 434.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 415.35 | 415.41 | 421.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:30:00 | 419.55 | 415.41 | 421.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 422.60 | 417.56 | 421.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:45:00 | 421.75 | 417.56 | 421.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 420.60 | 418.17 | 421.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:15:00 | 417.75 | 419.57 | 421.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:00:00 | 418.55 | 419.15 | 420.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 09:45:00 | 418.10 | 418.39 | 419.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 11:00:00 | 419.15 | 418.54 | 419.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 418.80 | 418.59 | 419.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:30:00 | 419.95 | 418.59 | 419.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 421.35 | 419.14 | 419.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 421.35 | 419.14 | 419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 420.70 | 419.45 | 419.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:45:00 | 420.65 | 419.45 | 419.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 419.70 | 419.50 | 419.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 429.00 | 421.48 | 420.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 429.00 | 421.48 | 420.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 432.20 | 426.75 | 423.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 433.45 | 435.22 | 432.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 433.45 | 435.22 | 432.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 433.45 | 435.22 | 432.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 10:00:00 | 433.45 | 435.22 | 432.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 436.00 | 435.38 | 433.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 11:45:00 | 437.40 | 435.55 | 433.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 430.55 | 434.26 | 433.26 | SL hit (close<static) qty=1.00 sl=432.90 alert=retest2 |

### Cycle 72 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 440.65 | 445.67 | 446.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 11:15:00 | 440.20 | 444.58 | 445.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 448.80 | 443.04 | 444.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 448.80 | 443.04 | 444.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 448.80 | 443.04 | 444.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 447.40 | 443.04 | 444.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 451.40 | 444.71 | 444.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:00:00 | 451.40 | 444.71 | 444.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 450.80 | 445.93 | 445.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 452.65 | 447.27 | 445.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 13:15:00 | 454.85 | 455.60 | 452.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 14:00:00 | 454.85 | 455.60 | 452.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 449.90 | 454.56 | 452.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 443.40 | 454.56 | 452.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 453.75 | 453.99 | 452.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:45:00 | 453.15 | 453.99 | 452.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 453.45 | 453.88 | 452.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:00:00 | 453.45 | 453.88 | 452.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 451.90 | 453.49 | 452.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:30:00 | 452.80 | 453.49 | 452.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 450.95 | 452.98 | 452.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:30:00 | 455.25 | 453.01 | 452.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 11:30:00 | 452.90 | 452.85 | 452.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 12:45:00 | 453.30 | 452.44 | 452.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 13:45:00 | 452.80 | 452.61 | 452.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 453.50 | 452.79 | 452.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:45:00 | 452.05 | 452.79 | 452.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 453.25 | 452.88 | 452.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 454.05 | 452.88 | 452.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 453.80 | 453.07 | 452.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-18 13:15:00 | 447.35 | 452.06 | 452.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 447.35 | 452.06 | 452.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 437.65 | 449.18 | 451.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 440.75 | 438.45 | 442.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 440.75 | 438.45 | 442.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 440.75 | 438.45 | 442.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:30:00 | 441.25 | 438.45 | 442.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 444.25 | 440.15 | 442.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:00:00 | 444.25 | 440.15 | 442.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 442.55 | 440.63 | 442.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:30:00 | 441.70 | 441.93 | 442.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 13:00:00 | 440.60 | 441.79 | 442.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 14:15:00 | 440.20 | 441.93 | 442.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 10:00:00 | 442.10 | 441.77 | 442.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 443.35 | 442.09 | 442.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 11:00:00 | 443.35 | 442.09 | 442.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 446.40 | 442.95 | 442.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 446.40 | 442.95 | 442.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 12:15:00 | 448.15 | 443.99 | 443.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 443.80 | 444.65 | 443.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 14:15:00 | 443.80 | 444.65 | 443.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 443.80 | 444.65 | 443.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 443.80 | 444.65 | 443.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 444.20 | 444.56 | 443.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 443.40 | 444.56 | 443.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 446.80 | 445.01 | 444.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:00:00 | 449.00 | 445.81 | 444.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 454.05 | 460.20 | 460.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 454.05 | 460.20 | 460.91 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 10:15:00 | 464.45 | 461.03 | 460.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 468.80 | 463.58 | 462.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 459.35 | 463.23 | 462.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 459.35 | 463.23 | 462.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 459.35 | 463.23 | 462.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:00:00 | 459.35 | 463.23 | 462.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 453.80 | 461.34 | 461.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 446.45 | 457.34 | 459.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 13:15:00 | 449.00 | 448.57 | 452.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 14:00:00 | 449.00 | 448.57 | 452.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 440.35 | 447.07 | 450.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 435.80 | 445.44 | 449.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 459.10 | 449.99 | 449.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 459.10 | 449.99 | 449.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 462.90 | 452.57 | 450.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 463.15 | 463.96 | 459.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 463.15 | 463.96 | 459.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 458.40 | 462.85 | 459.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 459.30 | 462.85 | 459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 469.00 | 464.08 | 460.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 472.20 | 464.71 | 460.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 469.30 | 466.57 | 462.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 470.00 | 467.96 | 463.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 491.70 | 495.26 | 495.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 491.70 | 495.26 | 495.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 10:15:00 | 489.85 | 494.18 | 495.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 489.20 | 488.48 | 490.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 14:00:00 | 489.20 | 488.48 | 490.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 479.85 | 483.77 | 486.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 484.00 | 483.77 | 486.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 483.45 | 483.34 | 485.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 485.60 | 483.34 | 485.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 491.90 | 485.06 | 486.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 491.90 | 485.06 | 486.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 490.20 | 486.08 | 486.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 492.00 | 486.08 | 486.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 492.00 | 487.27 | 486.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 524.90 | 494.79 | 490.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 478.85 | 502.87 | 498.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 478.85 | 502.87 | 498.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 478.85 | 502.87 | 498.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 480.35 | 502.87 | 498.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 461.15 | 494.53 | 495.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 416.50 | 478.92 | 488.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 455.20 | 451.95 | 464.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 455.20 | 451.95 | 464.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 482.85 | 460.53 | 465.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 482.85 | 460.53 | 465.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 481.50 | 464.73 | 466.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 473.80 | 466.33 | 467.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 14:15:00 | 472.50 | 468.48 | 468.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 472.50 | 468.48 | 468.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 473.95 | 469.57 | 468.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 477.55 | 479.04 | 476.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 477.55 | 479.04 | 476.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 477.55 | 479.04 | 476.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 479.00 | 479.04 | 476.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 479.70 | 478.92 | 476.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 479.15 | 478.92 | 476.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 476.45 | 478.62 | 477.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 476.45 | 478.62 | 477.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 477.40 | 478.37 | 477.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 486.45 | 478.37 | 477.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 480.20 | 485.45 | 486.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 480.20 | 485.45 | 486.04 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 490.15 | 484.54 | 484.26 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 479.40 | 483.79 | 484.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 475.25 | 481.54 | 483.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 468.25 | 466.56 | 469.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 468.25 | 466.56 | 469.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 473.20 | 468.26 | 469.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 475.70 | 468.26 | 469.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 473.20 | 469.25 | 469.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 474.10 | 469.25 | 469.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 473.15 | 470.67 | 470.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 474.50 | 472.98 | 471.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 477.00 | 477.56 | 474.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 477.00 | 477.56 | 474.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 477.00 | 477.56 | 474.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 477.00 | 477.56 | 474.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 490.80 | 490.42 | 487.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 488.20 | 490.42 | 487.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 488.45 | 490.03 | 487.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:45:00 | 489.20 | 490.03 | 487.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 492.80 | 490.58 | 488.33 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 489.15 | 490.73 | 490.94 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 501.55 | 492.88 | 491.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 505.45 | 495.39 | 493.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 498.15 | 501.22 | 498.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 498.15 | 501.22 | 498.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 498.15 | 501.22 | 498.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:00:00 | 498.15 | 501.22 | 498.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 498.55 | 500.69 | 498.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:15:00 | 496.95 | 500.69 | 498.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 495.95 | 499.74 | 498.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 495.95 | 499.74 | 498.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 497.35 | 499.26 | 498.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 494.50 | 499.26 | 498.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 500.15 | 499.18 | 498.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 501.60 | 499.18 | 498.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 12:15:00 | 497.05 | 499.15 | 498.44 | SL hit (close<static) qty=1.00 sl=497.40 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 494.10 | 503.37 | 504.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 489.70 | 500.64 | 502.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 493.15 | 492.81 | 497.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 493.15 | 492.81 | 497.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 486.55 | 491.27 | 494.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 470.20 | 490.03 | 493.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 485.70 | 487.29 | 490.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 498.30 | 490.61 | 491.62 | SL hit (close>static) qty=1.00 sl=495.80 alert=retest2 |

### Cycle 91 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 497.25 | 493.26 | 492.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 504.95 | 496.84 | 494.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 516.60 | 517.30 | 512.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 516.60 | 517.30 | 512.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 522.45 | 518.60 | 513.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:00:00 | 524.25 | 519.88 | 515.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 537.20 | 521.14 | 517.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 513.40 | 525.36 | 526.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 513.40 | 525.36 | 526.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 501.90 | 520.67 | 523.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 512.90 | 511.03 | 516.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 512.90 | 511.03 | 516.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 512.90 | 511.03 | 516.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:30:00 | 508.00 | 510.03 | 514.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:00:00 | 507.40 | 510.03 | 514.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 505.95 | 508.93 | 513.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 504.55 | 508.93 | 513.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 519.45 | 508.44 | 512.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 519.45 | 508.44 | 512.10 | SL hit (close>static) qty=1.00 sl=517.20 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 527.50 | 514.42 | 514.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 529.15 | 517.37 | 515.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 526.50 | 526.60 | 522.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 526.50 | 526.60 | 522.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 523.20 | 525.41 | 522.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 529.35 | 525.41 | 522.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:15:00 | 528.40 | 524.34 | 523.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:30:00 | 524.85 | 525.87 | 524.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:45:00 | 524.55 | 525.92 | 525.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 523.90 | 525.51 | 524.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 523.90 | 525.51 | 524.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 523.60 | 525.13 | 524.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 525.35 | 525.13 | 524.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 524.50 | 525.00 | 524.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 525.20 | 525.81 | 525.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 521.55 | 525.22 | 525.09 | SL hit (close<static) qty=1.00 sl=523.35 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 520.70 | 524.32 | 524.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 506.00 | 520.29 | 522.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 513.55 | 508.96 | 513.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 13:15:00 | 513.55 | 508.96 | 513.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 513.55 | 508.96 | 513.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 513.55 | 508.96 | 513.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 512.25 | 509.62 | 512.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 513.50 | 509.62 | 512.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 517.20 | 511.47 | 513.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:15:00 | 517.55 | 511.47 | 513.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 516.00 | 512.37 | 513.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 517.00 | 512.37 | 513.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 521.80 | 515.25 | 514.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 526.05 | 520.00 | 517.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 10:15:00 | 529.45 | 529.78 | 526.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 11:00:00 | 529.45 | 529.78 | 526.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 529.10 | 529.50 | 526.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:30:00 | 532.40 | 530.73 | 528.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 15:15:00 | 531.75 | 534.92 | 535.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 531.75 | 534.92 | 535.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 529.20 | 533.77 | 534.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 526.40 | 524.64 | 527.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 526.40 | 524.64 | 527.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 526.40 | 524.64 | 527.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 526.40 | 524.64 | 527.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 529.15 | 525.54 | 527.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 531.70 | 525.54 | 527.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 530.10 | 526.45 | 528.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 528.15 | 526.64 | 528.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:00:00 | 528.05 | 526.92 | 528.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:45:00 | 528.25 | 527.89 | 528.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 524.50 | 527.21 | 528.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 523.20 | 525.89 | 527.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 519.85 | 524.45 | 526.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 520.70 | 521.68 | 524.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 501.74 | 514.20 | 518.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 501.65 | 514.20 | 518.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 501.84 | 514.20 | 518.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 14:15:00 | 498.27 | 502.82 | 508.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 493.86 | 498.73 | 505.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 494.67 | 498.73 | 505.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 484.00 | 483.79 | 490.46 | SL hit (close>ema200) qty=0.50 sl=483.79 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 496.10 | 488.31 | 488.11 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 488.55 | 490.41 | 490.58 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 491.95 | 490.85 | 490.72 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 489.20 | 490.52 | 490.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 14:15:00 | 486.70 | 489.57 | 490.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 489.55 | 483.38 | 485.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 489.55 | 483.38 | 485.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 489.55 | 483.38 | 485.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 489.25 | 483.38 | 485.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 493.70 | 485.45 | 486.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 493.70 | 485.45 | 486.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 491.85 | 487.86 | 487.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 13:15:00 | 495.10 | 489.30 | 488.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 502.85 | 502.90 | 499.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:30:00 | 503.80 | 502.90 | 499.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 503.50 | 503.96 | 501.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 500.50 | 503.96 | 501.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 503.10 | 503.74 | 501.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 503.10 | 503.74 | 501.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 503.00 | 503.58 | 502.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:00:00 | 503.00 | 503.58 | 502.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 505.15 | 510.54 | 507.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 505.15 | 510.54 | 507.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 509.60 | 510.35 | 508.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:45:00 | 511.25 | 510.66 | 508.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 13:30:00 | 509.85 | 510.45 | 508.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 14:45:00 | 510.85 | 510.34 | 508.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 512.55 | 510.16 | 508.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 508.50 | 510.94 | 510.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 508.50 | 510.94 | 510.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 508.50 | 510.46 | 509.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 507.20 | 510.46 | 509.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 505.50 | 509.46 | 509.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 505.50 | 509.46 | 509.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 498.10 | 503.70 | 506.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 503.50 | 503.27 | 505.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:45:00 | 504.10 | 503.27 | 505.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 489.45 | 486.36 | 490.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 489.45 | 486.36 | 490.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 491.75 | 487.44 | 490.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 491.75 | 487.44 | 490.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 492.15 | 488.38 | 490.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 489.80 | 488.38 | 490.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 494.50 | 490.23 | 491.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 494.80 | 490.23 | 491.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 486.95 | 490.28 | 491.04 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 492.85 | 490.62 | 490.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 501.70 | 493.68 | 492.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 496.80 | 497.49 | 495.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 496.80 | 497.49 | 495.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 496.80 | 497.49 | 495.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 496.80 | 497.49 | 495.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 493.40 | 496.56 | 495.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 493.40 | 496.56 | 495.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 491.15 | 495.47 | 495.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 491.15 | 495.47 | 495.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 497.05 | 495.51 | 495.13 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 490.40 | 494.75 | 495.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 487.85 | 492.67 | 493.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 491.05 | 490.70 | 492.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 491.05 | 490.70 | 492.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 491.05 | 490.70 | 492.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 491.05 | 490.70 | 492.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 490.80 | 490.72 | 492.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:15:00 | 491.75 | 490.72 | 492.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 491.60 | 490.90 | 492.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 491.60 | 490.90 | 492.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 491.70 | 491.06 | 492.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 491.70 | 491.06 | 492.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 491.05 | 491.06 | 492.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 486.95 | 491.06 | 492.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 486.90 | 490.23 | 491.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 481.30 | 486.21 | 488.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:45:00 | 479.85 | 475.72 | 477.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 457.24 | 460.42 | 467.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 455.86 | 460.42 | 467.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 445.75 | 444.83 | 451.32 | SL hit (close>ema200) qty=0.50 sl=444.83 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 452.90 | 450.82 | 450.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 455.70 | 451.80 | 451.25 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 440.00 | 449.87 | 450.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 438.70 | 447.64 | 449.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 436.20 | 436.19 | 440.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 436.20 | 436.19 | 440.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 436.80 | 435.43 | 437.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 439.20 | 435.43 | 437.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 436.45 | 435.63 | 437.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 429.20 | 435.61 | 436.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 407.74 | 415.92 | 420.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 411.85 | 409.94 | 413.94 | SL hit (close>ema200) qty=0.50 sl=409.94 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 418.75 | 414.26 | 413.92 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 408.55 | 413.64 | 414.14 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 413.85 | 411.90 | 411.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 424.00 | 414.62 | 412.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 417.15 | 418.30 | 415.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 417.15 | 418.30 | 415.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 417.15 | 418.30 | 415.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 417.15 | 418.30 | 415.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 415.70 | 417.88 | 416.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 414.60 | 417.88 | 416.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 414.70 | 417.24 | 415.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:45:00 | 414.30 | 417.24 | 415.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 411.55 | 414.60 | 414.94 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 418.00 | 415.08 | 414.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 421.95 | 417.43 | 416.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 418.05 | 418.10 | 416.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 418.05 | 418.10 | 416.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 417.60 | 418.00 | 416.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 417.60 | 418.00 | 416.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 415.55 | 417.51 | 416.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 415.55 | 417.51 | 416.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 416.50 | 417.31 | 416.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 414.90 | 417.31 | 416.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 411.75 | 416.20 | 416.30 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 418.95 | 416.03 | 415.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 422.75 | 418.67 | 417.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 420.40 | 420.73 | 419.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:45:00 | 420.20 | 420.73 | 419.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 419.75 | 421.56 | 420.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 419.55 | 421.56 | 420.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 418.75 | 421.00 | 420.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 418.20 | 421.00 | 420.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 416.95 | 420.03 | 419.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 416.95 | 420.03 | 419.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 417.25 | 419.47 | 419.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 413.40 | 418.26 | 419.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 417.50 | 417.34 | 418.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 12:15:00 | 417.50 | 417.34 | 418.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 417.50 | 417.34 | 418.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 415.50 | 418.16 | 418.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 10:15:00 | 420.10 | 418.55 | 418.67 | SL hit (close>static) qty=1.00 sl=418.40 alert=retest2 |

### Cycle 115 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 420.25 | 418.97 | 418.83 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 417.25 | 418.63 | 418.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 416.80 | 418.14 | 418.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 416.65 | 415.62 | 416.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 10:15:00 | 416.65 | 415.62 | 416.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 416.65 | 415.62 | 416.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 415.90 | 415.62 | 416.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 415.00 | 415.50 | 416.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:30:00 | 416.60 | 415.50 | 416.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 415.50 | 414.69 | 415.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 417.65 | 414.69 | 415.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 416.70 | 415.10 | 415.78 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 417.00 | 416.27 | 416.18 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 412.80 | 415.68 | 415.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 411.95 | 414.93 | 415.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 410.25 | 408.55 | 410.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:45:00 | 410.15 | 408.55 | 410.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 410.35 | 408.91 | 410.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:45:00 | 408.15 | 409.04 | 410.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 411.55 | 409.17 | 410.08 | SL hit (close>static) qty=1.00 sl=411.25 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 384.20 | 382.27 | 382.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 386.20 | 383.06 | 382.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 11:15:00 | 385.10 | 385.61 | 384.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 11:15:00 | 385.10 | 385.61 | 384.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 385.10 | 385.61 | 384.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 385.10 | 385.61 | 384.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 387.50 | 385.99 | 384.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:15:00 | 388.30 | 385.99 | 384.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 10:00:00 | 388.55 | 392.40 | 390.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 382.80 | 390.48 | 389.56 | SL hit (close<static) qty=1.00 sl=384.60 alert=retest2 |

### Cycle 120 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 381.25 | 387.53 | 388.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 379.60 | 385.94 | 387.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 380.65 | 379.46 | 381.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 380.65 | 379.46 | 381.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 380.35 | 379.64 | 381.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 376.25 | 379.81 | 381.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 378.10 | 370.00 | 369.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 378.10 | 370.00 | 369.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 380.65 | 372.13 | 370.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 15:15:00 | 374.00 | 374.10 | 372.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:15:00 | 376.15 | 374.10 | 372.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 384.15 | 385.65 | 383.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 384.90 | 385.44 | 383.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 12:15:00 | 383.30 | 385.01 | 383.91 | SL hit (close<ema400) qty=1.00 sl=383.91 alert=retest1 |

### Cycle 122 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 377.60 | 382.36 | 382.92 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 386.40 | 382.46 | 382.00 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 378.00 | 382.55 | 383.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 15:15:00 | 374.00 | 378.26 | 380.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 374.85 | 373.51 | 376.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 374.85 | 373.51 | 376.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 374.85 | 373.51 | 376.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 376.40 | 373.51 | 376.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 375.85 | 373.98 | 376.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 375.85 | 373.98 | 376.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 377.70 | 374.72 | 376.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 377.70 | 374.72 | 376.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 377.35 | 375.25 | 376.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 378.70 | 375.25 | 376.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 380.50 | 377.18 | 377.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 387.00 | 380.24 | 378.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 379.65 | 382.79 | 380.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 379.65 | 382.79 | 380.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 379.65 | 382.79 | 380.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 382.00 | 382.79 | 380.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 370.70 | 384.52 | 385.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 370.70 | 384.52 | 385.47 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 383.20 | 379.93 | 379.68 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 378.40 | 380.04 | 380.19 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 11:15:00 | 381.75 | 380.18 | 380.11 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 377.95 | 379.81 | 379.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 371.15 | 377.75 | 378.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 360.25 | 360.21 | 365.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 360.25 | 360.21 | 365.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 363.30 | 360.79 | 363.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 357.50 | 360.71 | 362.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:00:00 | 358.40 | 356.25 | 358.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 355.50 | 358.07 | 358.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 357.25 | 357.76 | 358.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 357.80 | 357.77 | 358.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:45:00 | 357.10 | 357.77 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 358.95 | 357.73 | 358.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 358.95 | 357.73 | 358.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 360.95 | 358.37 | 358.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 360.95 | 358.37 | 358.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 360.20 | 358.74 | 358.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 360.20 | 358.74 | 358.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 363.35 | 359.66 | 359.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 361.70 | 361.79 | 360.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 09:15:00 | 362.65 | 361.79 | 360.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 362.40 | 361.91 | 360.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:00:00 | 364.35 | 362.40 | 361.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 364.80 | 366.27 | 365.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 359.95 | 364.39 | 364.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 359.95 | 364.39 | 364.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 358.20 | 361.33 | 362.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 13:15:00 | 363.05 | 360.35 | 361.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 13:15:00 | 363.05 | 360.35 | 361.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 363.05 | 360.35 | 361.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:30:00 | 364.05 | 360.35 | 361.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 363.80 | 361.04 | 361.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:30:00 | 364.70 | 361.04 | 361.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 09:15:00 | 373.35 | 363.81 | 363.06 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 355.20 | 364.06 | 364.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 11:15:00 | 353.50 | 361.95 | 363.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 360.55 | 360.18 | 361.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 10:15:00 | 360.55 | 360.18 | 361.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 360.55 | 360.18 | 361.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 360.55 | 360.18 | 361.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 362.65 | 360.72 | 361.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 362.45 | 360.72 | 361.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 365.90 | 361.75 | 362.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 365.90 | 361.75 | 362.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 15:15:00 | 363.55 | 362.40 | 362.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 366.55 | 363.23 | 362.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 381.10 | 381.36 | 376.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 381.10 | 381.36 | 376.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 378.65 | 380.65 | 378.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 378.75 | 380.65 | 378.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 377.25 | 379.97 | 378.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 377.05 | 379.97 | 378.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 375.20 | 379.02 | 377.93 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 372.65 | 377.04 | 377.18 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 378.60 | 377.04 | 376.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 13:15:00 | 379.65 | 377.58 | 377.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 379.20 | 379.71 | 378.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 379.20 | 379.71 | 378.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 378.85 | 379.54 | 378.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 385.60 | 379.16 | 378.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 398.85 | 401.79 | 401.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 398.85 | 401.79 | 401.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 397.20 | 399.39 | 400.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 398.95 | 397.43 | 398.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 398.95 | 397.43 | 398.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 398.95 | 397.43 | 398.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 398.95 | 397.43 | 398.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 398.90 | 397.72 | 398.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:15:00 | 399.05 | 397.72 | 398.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 398.85 | 397.95 | 398.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 399.05 | 397.95 | 398.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 397.90 | 397.94 | 398.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 398.50 | 397.94 | 398.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 400.20 | 397.99 | 398.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 400.45 | 397.99 | 398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 400.10 | 398.41 | 398.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 401.30 | 398.41 | 398.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 398.85 | 397.60 | 398.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:45:00 | 398.50 | 397.60 | 398.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 397.30 | 397.54 | 398.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 399.95 | 397.54 | 398.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 400.35 | 398.10 | 398.28 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 399.45 | 398.51 | 398.43 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 397.65 | 398.34 | 398.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 393.15 | 397.25 | 397.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 396.60 | 396.50 | 397.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 13:00:00 | 396.60 | 396.50 | 397.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 395.30 | 396.26 | 397.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 396.25 | 396.26 | 397.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 396.80 | 396.36 | 397.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 396.80 | 396.36 | 397.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 398.00 | 396.69 | 397.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 397.55 | 397.11 | 397.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 397.45 | 397.18 | 397.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 395.20 | 397.06 | 397.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 397.55 | 397.43 | 397.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 397.55 | 397.43 | 397.42 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 14:15:00 | 397.25 | 397.40 | 397.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 385.55 | 394.94 | 396.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 378.50 | 376.76 | 383.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 378.50 | 376.76 | 383.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 378.50 | 377.40 | 382.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 376.10 | 379.93 | 381.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 375.95 | 378.63 | 380.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 388.25 | 379.53 | 380.28 | SL hit (close>static) qty=1.00 sl=385.50 alert=retest2 |

### Cycle 143 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 388.50 | 381.32 | 381.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 391.15 | 383.29 | 381.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 390.65 | 396.29 | 393.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 390.65 | 396.29 | 393.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 390.65 | 396.29 | 393.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 391.80 | 396.29 | 393.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 392.70 | 395.57 | 393.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 394.10 | 395.57 | 393.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 12:15:00 | 398.45 | 399.20 | 399.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 398.45 | 399.20 | 399.28 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 400.25 | 399.41 | 399.37 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 398.85 | 399.30 | 399.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 15:15:00 | 398.50 | 399.14 | 399.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 399.45 | 399.12 | 399.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 399.45 | 399.12 | 399.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 399.45 | 399.12 | 399.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 398.65 | 399.12 | 399.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 400.80 | 399.46 | 399.36 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 394.30 | 398.69 | 399.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 391.25 | 394.99 | 396.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 390.40 | 387.70 | 390.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 390.40 | 387.70 | 390.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 390.40 | 387.70 | 390.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 390.40 | 387.70 | 390.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 384.95 | 387.15 | 389.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 383.20 | 387.15 | 389.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 383.35 | 385.48 | 388.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 383.35 | 385.38 | 387.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 381.15 | 384.70 | 386.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 383.40 | 380.92 | 382.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 383.40 | 380.92 | 382.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 383.45 | 381.43 | 382.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 388.50 | 383.93 | 383.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 388.50 | 383.93 | 383.63 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 380.60 | 383.54 | 383.70 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 392.55 | 384.70 | 383.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 398.95 | 393.27 | 389.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 394.55 | 395.45 | 392.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 394.55 | 395.45 | 392.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 403.55 | 404.73 | 403.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 403.55 | 404.73 | 403.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 401.40 | 403.89 | 403.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 401.40 | 403.89 | 403.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 403.05 | 403.72 | 403.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 405.00 | 403.61 | 403.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 15:15:00 | 405.40 | 406.87 | 407.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 405.40 | 406.87 | 407.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 400.00 | 405.49 | 406.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 401.35 | 400.31 | 402.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 401.35 | 400.31 | 402.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 401.35 | 400.31 | 402.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 401.55 | 400.31 | 402.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 401.95 | 400.64 | 402.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 401.85 | 400.64 | 402.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 403.00 | 401.58 | 402.24 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 405.30 | 402.63 | 402.62 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 400.95 | 402.51 | 402.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 397.75 | 399.99 | 401.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 400.70 | 399.16 | 400.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 400.70 | 399.16 | 400.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 400.70 | 399.16 | 400.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 400.70 | 399.16 | 400.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 398.90 | 399.11 | 399.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 397.80 | 399.05 | 399.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 397.90 | 398.75 | 399.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 398.35 | 398.92 | 399.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 398.60 | 398.93 | 399.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 396.90 | 398.53 | 399.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 396.70 | 398.27 | 398.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 400.45 | 398.71 | 399.04 | SL hit (close>static) qty=1.00 sl=399.20 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 400.10 | 399.39 | 399.29 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 394.50 | 398.40 | 398.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 392.65 | 396.43 | 397.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 393.85 | 393.34 | 395.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 393.85 | 393.34 | 395.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 395.00 | 394.11 | 394.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 395.10 | 394.11 | 394.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 394.30 | 394.15 | 394.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 394.80 | 394.15 | 394.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 397.00 | 394.72 | 395.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 397.00 | 394.72 | 395.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 396.00 | 394.98 | 395.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:15:00 | 394.55 | 394.98 | 395.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 398.90 | 395.59 | 395.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 398.90 | 395.59 | 395.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 402.25 | 398.74 | 397.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 401.00 | 401.40 | 399.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 15:00:00 | 401.00 | 401.40 | 399.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 400.20 | 401.55 | 400.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 400.20 | 401.55 | 400.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 399.20 | 401.08 | 400.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 399.20 | 401.08 | 400.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 400.00 | 400.86 | 400.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 404.20 | 400.86 | 400.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 402.65 | 403.63 | 402.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 402.90 | 403.63 | 402.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 402.40 | 403.39 | 402.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 402.40 | 403.39 | 402.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 402.65 | 403.24 | 402.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 401.15 | 403.24 | 402.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 398.85 | 402.36 | 401.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 398.85 | 402.36 | 401.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 398.50 | 401.59 | 401.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 395.35 | 400.34 | 401.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 391.30 | 391.25 | 393.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 391.30 | 391.25 | 393.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 393.70 | 391.82 | 393.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 393.70 | 391.82 | 393.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 393.70 | 392.20 | 393.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 393.70 | 392.20 | 393.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 394.45 | 392.65 | 393.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 394.45 | 392.65 | 393.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 394.05 | 392.93 | 393.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 394.00 | 392.93 | 393.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 393.45 | 393.03 | 393.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 392.60 | 393.29 | 393.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 391.80 | 391.98 | 392.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 391.30 | 388.98 | 388.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 391.30 | 388.98 | 388.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 392.95 | 390.10 | 389.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 391.45 | 391.97 | 390.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 391.45 | 391.97 | 390.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 391.45 | 391.97 | 390.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 391.15 | 391.97 | 390.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 392.60 | 392.69 | 391.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:45:00 | 393.05 | 392.69 | 391.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 391.85 | 392.48 | 391.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 391.55 | 392.48 | 391.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 391.90 | 392.36 | 391.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:30:00 | 391.25 | 392.36 | 391.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 392.00 | 392.29 | 391.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 393.35 | 392.29 | 391.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 393.55 | 392.54 | 391.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:15:00 | 394.00 | 392.43 | 392.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 394.00 | 394.24 | 393.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:00:00 | 394.00 | 394.19 | 393.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 394.55 | 394.26 | 393.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 395.25 | 394.69 | 393.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 395.25 | 394.69 | 393.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 392.50 | 394.25 | 393.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 392.50 | 394.25 | 393.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 392.70 | 393.94 | 393.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:15:00 | 392.30 | 393.94 | 393.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 391.05 | 392.92 | 393.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 391.05 | 392.92 | 393.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 389.85 | 391.46 | 392.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 388.05 | 387.87 | 389.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 388.05 | 387.87 | 389.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 385.50 | 383.82 | 384.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 385.50 | 383.82 | 384.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 384.95 | 384.04 | 384.50 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 387.70 | 385.29 | 384.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 15:15:00 | 388.00 | 385.83 | 385.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 384.25 | 385.51 | 385.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 384.25 | 385.51 | 385.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 384.25 | 385.51 | 385.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 384.25 | 385.51 | 385.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 383.00 | 385.01 | 384.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 383.00 | 385.01 | 384.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 384.10 | 384.83 | 384.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 382.30 | 383.94 | 384.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 383.55 | 382.33 | 383.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 383.55 | 382.33 | 383.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 383.55 | 382.33 | 383.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 383.55 | 382.33 | 383.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 383.00 | 382.46 | 383.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 382.00 | 382.46 | 383.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 381.85 | 382.46 | 383.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 382.65 | 382.49 | 382.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 384.70 | 383.33 | 383.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 384.70 | 383.33 | 383.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 385.90 | 384.19 | 383.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 384.20 | 384.48 | 383.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 13:15:00 | 384.20 | 384.48 | 383.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 384.20 | 384.48 | 383.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 384.20 | 384.48 | 383.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 386.60 | 384.91 | 384.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:45:00 | 387.20 | 385.89 | 384.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 387.35 | 386.24 | 385.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 387.15 | 386.49 | 385.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:30:00 | 387.05 | 386.78 | 386.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 385.85 | 386.50 | 386.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 387.50 | 386.50 | 386.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:00:00 | 387.70 | 386.38 | 386.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 387.00 | 387.08 | 386.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 386.60 | 387.08 | 386.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 386.85 | 387.03 | 386.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 386.30 | 387.03 | 386.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 386.85 | 387.00 | 386.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 386.70 | 387.00 | 386.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 386.50 | 386.90 | 386.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 385.95 | 386.90 | 386.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 385.75 | 386.67 | 386.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 385.75 | 386.67 | 386.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 388.00 | 386.93 | 386.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 391.10 | 387.52 | 387.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 384.85 | 388.27 | 388.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 384.85 | 388.27 | 388.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 382.35 | 385.74 | 387.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 383.15 | 382.48 | 384.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 383.15 | 382.48 | 384.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 381.70 | 380.22 | 381.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 381.70 | 380.22 | 381.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 381.65 | 380.50 | 381.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 381.65 | 380.50 | 381.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 382.05 | 380.81 | 381.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 382.05 | 380.81 | 381.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 383.65 | 381.38 | 381.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 383.65 | 381.38 | 381.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 383.80 | 381.86 | 382.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 381.25 | 381.86 | 382.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 376.65 | 377.41 | 378.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 372.20 | 375.58 | 377.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:00:00 | 372.50 | 374.62 | 376.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 378.05 | 376.31 | 376.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 378.05 | 376.31 | 376.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 379.85 | 377.02 | 376.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 375.65 | 377.18 | 376.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 375.65 | 377.18 | 376.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 375.65 | 377.18 | 376.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 11:00:00 | 377.00 | 377.15 | 376.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 377.00 | 376.99 | 376.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 376.85 | 377.03 | 376.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:45:00 | 376.80 | 376.94 | 376.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 376.50 | 376.85 | 376.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 376.90 | 376.85 | 376.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 376.25 | 376.73 | 376.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 376.10 | 376.73 | 376.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 375.80 | 376.54 | 376.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 375.80 | 376.54 | 376.62 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 379.45 | 377.15 | 376.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 380.30 | 378.97 | 377.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 379.15 | 379.33 | 378.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 379.15 | 379.33 | 378.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 378.95 | 379.25 | 378.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 379.75 | 379.25 | 378.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 377.00 | 378.80 | 378.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 377.00 | 378.80 | 378.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 377.95 | 378.63 | 378.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 377.55 | 378.63 | 378.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 380.15 | 378.94 | 378.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 381.00 | 378.94 | 378.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 384.85 | 385.63 | 385.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 384.85 | 385.63 | 385.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 380.05 | 384.15 | 384.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 378.85 | 376.52 | 378.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 378.85 | 376.52 | 378.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 378.85 | 376.52 | 378.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 378.85 | 376.52 | 378.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 378.60 | 376.94 | 378.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:15:00 | 379.50 | 376.94 | 378.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 380.10 | 377.57 | 378.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 380.10 | 377.57 | 378.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 380.55 | 378.16 | 379.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 380.50 | 378.16 | 379.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 375.40 | 374.49 | 375.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 375.40 | 374.49 | 375.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 375.20 | 374.58 | 375.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 374.55 | 374.58 | 375.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 374.90 | 374.65 | 375.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 375.75 | 374.65 | 375.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 377.70 | 375.26 | 375.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 377.70 | 375.26 | 375.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 376.20 | 375.45 | 375.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 375.90 | 375.56 | 375.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:00:00 | 376.00 | 375.56 | 375.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 375.80 | 375.66 | 375.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 375.30 | 375.58 | 375.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 376.00 | 375.54 | 375.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 376.05 | 375.78 | 375.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 376.05 | 375.78 | 375.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 377.85 | 376.32 | 376.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 379.65 | 380.49 | 378.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 379.65 | 380.49 | 378.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 389.10 | 391.05 | 389.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 389.10 | 391.05 | 389.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 386.90 | 390.22 | 389.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 386.90 | 390.22 | 389.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 387.35 | 389.65 | 389.34 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 386.95 | 389.11 | 389.12 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 390.80 | 388.93 | 388.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 391.10 | 389.36 | 389.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 394.10 | 394.17 | 393.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:15:00 | 396.30 | 394.17 | 393.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 15:15:00 | 394.95 | 395.55 | 394.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 394.95 | 395.43 | 394.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 396.15 | 395.43 | 394.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 396.40 | 395.62 | 394.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 394.30 | 395.40 | 394.89 | SL hit (close<ema400) qty=1.00 sl=394.89 alert=retest1 |

### Cycle 172 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 391.10 | 396.40 | 396.54 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 396.10 | 395.25 | 395.23 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 394.55 | 395.18 | 395.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 393.25 | 394.46 | 394.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 394.60 | 394.49 | 394.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 394.60 | 394.49 | 394.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 394.10 | 394.41 | 394.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 394.30 | 394.41 | 394.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 394.55 | 394.44 | 394.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 394.75 | 394.44 | 394.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 394.40 | 394.43 | 394.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 394.85 | 394.43 | 394.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 393.70 | 394.22 | 394.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 394.95 | 394.22 | 394.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 393.30 | 394.03 | 394.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 392.45 | 393.78 | 394.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 392.40 | 393.74 | 394.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 392.45 | 393.48 | 393.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 390.95 | 393.35 | 393.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 388.00 | 389.34 | 390.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 389.90 | 389.34 | 390.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 389.85 | 389.45 | 390.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:45:00 | 388.30 | 389.19 | 390.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 392.80 | 390.09 | 390.29 | SL hit (close>static) qty=1.00 sl=391.40 alert=retest2 |

### Cycle 175 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 391.30 | 390.57 | 390.49 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 388.40 | 390.05 | 390.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 383.00 | 388.61 | 389.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 15:15:00 | 382.25 | 382.16 | 384.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 388.00 | 382.16 | 384.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 386.50 | 383.03 | 384.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 386.45 | 383.03 | 384.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 387.10 | 383.84 | 384.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 387.10 | 383.84 | 384.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 384.60 | 384.63 | 384.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 383.85 | 384.58 | 384.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 385.85 | 383.63 | 384.03 | SL hit (close>static) qty=1.00 sl=385.50 alert=retest2 |

### Cycle 177 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 385.60 | 384.30 | 384.18 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 382.60 | 384.05 | 384.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 380.90 | 382.00 | 382.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 383.75 | 381.95 | 382.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 383.75 | 381.95 | 382.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 383.75 | 381.95 | 382.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 383.75 | 381.95 | 382.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 384.65 | 382.49 | 382.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 384.65 | 382.49 | 382.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 384.45 | 382.88 | 382.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 386.55 | 384.53 | 383.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 386.25 | 386.41 | 385.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:45:00 | 386.60 | 386.41 | 385.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 385.90 | 386.71 | 385.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 386.00 | 386.71 | 385.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 388.60 | 387.09 | 386.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 390.05 | 387.41 | 386.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:00:00 | 388.95 | 387.72 | 386.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 390.85 | 393.40 | 393.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 390.85 | 393.40 | 393.74 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 397.75 | 393.97 | 393.92 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 382.60 | 392.97 | 393.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 379.30 | 387.00 | 388.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 374.80 | 374.67 | 378.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 374.80 | 374.67 | 378.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 377.90 | 375.85 | 377.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 377.70 | 375.85 | 377.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 380.65 | 376.81 | 377.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 380.65 | 376.81 | 377.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 382.10 | 378.62 | 378.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 382.85 | 381.28 | 380.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 384.95 | 385.62 | 384.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 13:00:00 | 384.95 | 385.62 | 384.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 383.85 | 385.27 | 384.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 383.85 | 385.27 | 384.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 383.40 | 384.89 | 383.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 383.40 | 384.89 | 383.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 383.30 | 384.58 | 383.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 384.60 | 384.58 | 383.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 385.55 | 384.66 | 384.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 384.20 | 384.66 | 384.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 385.00 | 384.91 | 384.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 384.65 | 384.91 | 384.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 385.10 | 386.88 | 386.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 384.85 | 386.88 | 386.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 386.00 | 386.70 | 386.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 384.80 | 386.70 | 386.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 387.05 | 386.77 | 386.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 386.10 | 386.77 | 386.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 387.05 | 386.83 | 386.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 386.45 | 386.83 | 386.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 386.40 | 386.74 | 386.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 387.10 | 386.74 | 386.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 384.10 | 386.21 | 386.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 384.10 | 386.21 | 386.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 383.65 | 385.70 | 385.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 380.80 | 384.72 | 385.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 380.75 | 380.42 | 382.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:45:00 | 380.95 | 380.42 | 382.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 375.00 | 372.38 | 374.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 375.00 | 372.38 | 374.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 376.00 | 373.10 | 374.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 376.00 | 373.10 | 374.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 376.50 | 373.78 | 374.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 376.70 | 373.78 | 374.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 377.50 | 375.31 | 375.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 378.00 | 376.95 | 376.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 376.60 | 377.09 | 376.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 376.60 | 377.09 | 376.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 377.40 | 377.15 | 376.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 377.65 | 377.25 | 376.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 375.25 | 376.69 | 376.46 | SL hit (close<static) qty=1.00 sl=376.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 374.15 | 376.95 | 377.25 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 379.00 | 376.85 | 376.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 12:15:00 | 380.00 | 378.38 | 377.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 376.55 | 378.51 | 377.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 376.55 | 378.51 | 377.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 376.55 | 378.51 | 377.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 376.55 | 378.51 | 377.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 376.85 | 378.18 | 377.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 378.15 | 378.18 | 377.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 376.10 | 377.71 | 377.70 | SL hit (close<static) qty=1.00 sl=376.20 alert=retest2 |

### Cycle 188 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 377.20 | 377.61 | 377.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 374.85 | 376.96 | 377.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 378.60 | 377.29 | 377.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 378.60 | 377.29 | 377.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 378.60 | 377.29 | 377.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 378.60 | 377.29 | 377.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 379.75 | 377.78 | 377.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 380.35 | 378.60 | 378.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 383.60 | 383.93 | 382.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 12:00:00 | 383.60 | 383.93 | 382.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 383.80 | 383.83 | 382.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 383.05 | 383.83 | 382.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 383.50 | 383.76 | 382.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 383.10 | 383.76 | 382.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 381.90 | 383.33 | 382.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 380.85 | 383.33 | 382.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 383.20 | 383.30 | 382.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 382.00 | 383.30 | 382.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 384.35 | 383.51 | 382.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 383.30 | 383.51 | 382.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 382.55 | 383.70 | 383.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 382.55 | 383.70 | 383.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 380.55 | 383.07 | 383.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 379.30 | 382.32 | 382.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 382.15 | 381.88 | 382.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 15:15:00 | 382.15 | 381.88 | 382.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 382.15 | 381.88 | 382.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 382.60 | 381.88 | 382.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 383.80 | 382.26 | 382.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 384.05 | 382.26 | 382.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 384.85 | 382.78 | 382.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 385.60 | 384.40 | 383.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 383.95 | 384.75 | 384.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 383.95 | 384.75 | 384.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 383.95 | 384.75 | 384.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 383.95 | 384.75 | 384.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 384.55 | 384.71 | 384.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 384.55 | 384.71 | 384.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 385.30 | 384.83 | 384.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 385.00 | 384.83 | 384.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 384.20 | 384.70 | 384.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 384.20 | 384.70 | 384.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 385.10 | 384.78 | 384.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 385.30 | 384.98 | 384.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 385.80 | 384.98 | 384.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 386.35 | 384.99 | 384.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 386.10 | 385.12 | 384.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 384.70 | 385.04 | 384.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:00:00 | 384.70 | 385.04 | 384.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 385.10 | 385.05 | 384.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:15:00 | 385.95 | 385.12 | 384.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 397.25 | 400.26 | 400.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 397.25 | 400.26 | 400.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 396.65 | 399.54 | 400.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 402.00 | 399.15 | 399.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 402.00 | 399.15 | 399.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 402.00 | 399.15 | 399.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 402.80 | 399.15 | 399.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 402.05 | 399.73 | 399.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 402.95 | 399.73 | 399.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 401.60 | 400.11 | 399.94 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 399.45 | 399.77 | 399.80 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 400.90 | 399.83 | 399.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 400.95 | 400.15 | 399.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 426.10 | 426.17 | 420.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 426.10 | 426.17 | 420.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 426.15 | 428.57 | 426.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 426.15 | 428.57 | 426.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 426.35 | 428.12 | 426.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 424.80 | 428.12 | 426.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 425.20 | 427.54 | 426.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 425.20 | 427.54 | 426.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 424.90 | 427.01 | 426.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 424.10 | 427.01 | 426.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 425.65 | 426.74 | 426.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:15:00 | 424.15 | 426.74 | 426.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 424.15 | 426.22 | 425.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 425.00 | 426.22 | 425.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 425.95 | 426.17 | 425.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 423.80 | 426.17 | 425.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 424.10 | 425.75 | 425.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 424.10 | 425.75 | 425.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 422.20 | 425.04 | 425.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 420.65 | 424.16 | 424.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 426.20 | 422.57 | 423.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 426.20 | 422.57 | 423.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 426.20 | 422.57 | 423.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 426.20 | 422.57 | 423.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 430.50 | 424.16 | 424.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:45:00 | 430.80 | 424.16 | 424.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 429.75 | 425.28 | 424.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 432.75 | 428.35 | 426.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 429.35 | 429.67 | 427.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 429.35 | 429.67 | 427.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 429.45 | 429.66 | 428.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 427.85 | 429.66 | 428.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 429.00 | 429.53 | 428.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 440.40 | 429.53 | 428.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 437.50 | 431.12 | 429.19 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 430.35 | 430.86 | 430.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 423.30 | 428.97 | 429.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 424.90 | 417.61 | 420.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 424.90 | 417.61 | 420.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 424.90 | 417.61 | 420.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 424.90 | 417.61 | 420.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 423.00 | 418.69 | 420.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 421.65 | 418.69 | 420.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 423.00 | 421.16 | 421.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 423.00 | 421.16 | 421.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 424.80 | 421.89 | 421.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 420.95 | 421.70 | 421.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 10:15:00 | 420.95 | 421.70 | 421.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 420.95 | 421.70 | 421.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 420.25 | 421.70 | 421.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 415.75 | 420.51 | 420.91 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 10:15:00 | 422.90 | 420.79 | 420.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 433.65 | 424.44 | 422.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 445.25 | 449.02 | 441.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 445.25 | 449.02 | 441.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 439.40 | 447.09 | 441.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 439.40 | 447.09 | 441.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 438.10 | 445.29 | 441.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 438.10 | 445.29 | 441.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 438.60 | 443.96 | 440.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 440.80 | 442.60 | 440.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 436.00 | 440.87 | 440.31 | SL hit (close<static) qty=1.00 sl=437.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 435.75 | 439.84 | 439.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 427.95 | 437.46 | 438.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 423.85 | 421.03 | 426.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 423.85 | 421.03 | 426.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 422.15 | 421.54 | 426.00 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 438.00 | 428.07 | 427.22 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 429.60 | 430.26 | 430.32 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 432.70 | 430.65 | 430.48 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 429.80 | 430.96 | 431.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 422.15 | 428.97 | 430.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 419.60 | 413.39 | 416.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 419.60 | 413.39 | 416.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 419.60 | 413.39 | 416.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 419.60 | 413.39 | 416.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 420.15 | 414.74 | 417.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 419.30 | 414.74 | 417.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 422.20 | 419.11 | 418.75 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 415.05 | 418.91 | 419.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 414.15 | 417.96 | 418.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 418.20 | 417.45 | 418.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 418.20 | 417.45 | 418.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 418.20 | 417.45 | 418.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 418.20 | 417.45 | 418.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 415.95 | 417.27 | 418.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 415.40 | 417.27 | 418.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 415.60 | 416.91 | 417.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 415.70 | 416.98 | 417.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 415.60 | 416.95 | 417.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 424.00 | 418.14 | 418.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 424.00 | 418.14 | 418.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 425.80 | 422.10 | 420.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 433.70 | 435.22 | 431.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:45:00 | 431.55 | 435.22 | 431.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 430.30 | 434.24 | 431.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 429.75 | 434.24 | 431.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 432.15 | 433.82 | 431.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 432.90 | 433.82 | 431.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:00:00 | 432.85 | 433.62 | 431.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 433.00 | 433.73 | 431.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 434.15 | 433.73 | 431.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 432.75 | 433.53 | 431.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 428.75 | 433.53 | 431.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 427.65 | 432.36 | 431.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 427.65 | 432.36 | 431.47 | SL hit (close<static) qty=1.00 sl=429.75 alert=retest2 |

### Cycle 210 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 426.30 | 430.68 | 430.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 424.80 | 429.48 | 430.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 426.95 | 425.92 | 427.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 426.95 | 425.92 | 427.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 437.15 | 428.09 | 428.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 437.15 | 428.09 | 428.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 10:15:00 | 435.95 | 429.66 | 429.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 450.00 | 436.78 | 433.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 445.45 | 447.20 | 441.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 447.60 | 447.20 | 441.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 442.90 | 446.34 | 441.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 441.90 | 446.34 | 441.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 443.65 | 445.80 | 442.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 444.45 | 445.80 | 442.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 441.25 | 444.52 | 442.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 441.85 | 444.52 | 442.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 440.75 | 443.77 | 441.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 438.95 | 443.77 | 441.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 441.50 | 443.31 | 441.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 443.35 | 443.31 | 441.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 437.85 | 442.48 | 441.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:00:00 | 437.85 | 442.48 | 441.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 436.30 | 441.25 | 441.30 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 444.35 | 440.89 | 440.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 446.40 | 442.70 | 441.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 446.00 | 446.39 | 444.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 446.00 | 446.39 | 444.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 459.80 | 464.15 | 460.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 459.55 | 464.15 | 460.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 461.05 | 463.53 | 460.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:15:00 | 459.60 | 463.53 | 460.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 457.40 | 462.31 | 460.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:45:00 | 457.50 | 462.31 | 460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 456.50 | 461.15 | 460.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 456.50 | 461.15 | 460.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 459.00 | 460.61 | 459.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 461.40 | 460.61 | 459.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 461.95 | 460.88 | 460.16 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 453.25 | 459.99 | 460.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 11:15:00 | 450.55 | 457.09 | 458.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 15:15:00 | 457.20 | 455.88 | 457.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 15:15:00 | 457.20 | 455.88 | 457.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 457.20 | 455.88 | 457.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 454.75 | 455.88 | 457.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 455.45 | 455.79 | 457.34 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 467.30 | 457.57 | 457.38 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 454.05 | 460.42 | 460.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 443.35 | 454.74 | 457.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 445.20 | 443.83 | 446.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 445.20 | 443.83 | 446.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 445.20 | 443.83 | 446.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 446.25 | 443.83 | 446.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 447.40 | 444.54 | 446.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 447.40 | 444.54 | 446.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 444.65 | 444.57 | 446.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:45:00 | 447.50 | 444.57 | 446.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 457.85 | 447.31 | 447.20 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 444.05 | 449.24 | 449.63 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 457.50 | 450.58 | 449.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 459.10 | 453.34 | 451.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 09:15:00 | 449.75 | 458.38 | 456.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 449.75 | 458.38 | 456.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 449.75 | 458.38 | 456.74 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 11:15:00 | 449.45 | 455.19 | 455.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 13:15:00 | 448.15 | 452.68 | 454.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 09:15:00 | 452.60 | 451.64 | 453.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 452.60 | 451.64 | 453.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 452.60 | 451.64 | 453.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 452.60 | 451.64 | 453.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 455.10 | 452.34 | 453.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:45:00 | 455.40 | 452.34 | 453.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 452.95 | 452.46 | 453.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 449.05 | 450.48 | 452.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 438.60 | 436.60 | 436.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 438.60 | 436.60 | 436.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 441.50 | 437.55 | 436.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 12:15:00 | 442.90 | 443.78 | 442.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 13:00:00 | 442.90 | 443.78 | 442.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 445.25 | 444.33 | 442.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 11:45:00 | 448.75 | 445.29 | 443.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 472.80 | 476.30 | 476.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 472.80 | 476.30 | 476.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 14:15:00 | 470.65 | 473.16 | 474.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 11:15:00 | 472.05 | 471.27 | 472.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 472.05 | 471.27 | 472.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 461.50 | 469.32 | 471.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:30:00 | 472.50 | 469.32 | 471.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-13 11:30:00 | 228.20 | 2023-06-13 15:15:00 | 229.05 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-06-13 12:30:00 | 228.25 | 2023-06-13 15:15:00 | 229.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2023-06-14 11:15:00 | 227.85 | 2023-06-14 13:15:00 | 229.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-06-14 12:15:00 | 228.20 | 2023-06-14 13:15:00 | 229.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-07-04 11:15:00 | 232.35 | 2023-07-05 11:15:00 | 230.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-07-05 10:00:00 | 232.15 | 2023-07-13 09:15:00 | 232.20 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2023-07-06 09:15:00 | 232.50 | 2023-07-13 09:15:00 | 232.20 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-07-17 13:45:00 | 230.30 | 2023-07-24 10:15:00 | 231.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-07-17 14:45:00 | 230.00 | 2023-07-24 10:15:00 | 231.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-07-18 10:15:00 | 230.50 | 2023-07-24 10:15:00 | 231.15 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-07-18 11:00:00 | 230.50 | 2023-07-24 10:15:00 | 231.15 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-07-21 14:30:00 | 228.60 | 2023-07-24 10:15:00 | 231.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-07-26 09:15:00 | 230.75 | 2023-07-26 12:15:00 | 229.50 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-07-26 12:15:00 | 230.15 | 2023-07-26 12:15:00 | 229.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-07-31 12:15:00 | 228.70 | 2023-08-01 09:15:00 | 231.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-08-11 14:30:00 | 235.05 | 2023-08-14 09:15:00 | 232.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-08-11 15:15:00 | 235.30 | 2023-08-14 09:15:00 | 232.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2023-08-17 11:45:00 | 231.35 | 2023-08-22 12:15:00 | 231.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2023-08-18 09:15:00 | 228.70 | 2023-08-22 12:15:00 | 231.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-09-13 11:30:00 | 275.20 | 2023-09-21 15:15:00 | 280.30 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2023-09-13 12:00:00 | 276.35 | 2023-09-21 15:15:00 | 280.30 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2023-09-27 11:15:00 | 291.15 | 2023-10-04 09:15:00 | 289.75 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-10-06 11:15:00 | 288.90 | 2023-10-10 09:15:00 | 291.65 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-10-06 12:45:00 | 289.05 | 2023-10-10 09:15:00 | 291.65 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-10-06 14:00:00 | 289.30 | 2023-10-10 09:15:00 | 291.65 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-10-19 11:15:00 | 313.35 | 2023-10-20 12:15:00 | 310.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-10-19 12:00:00 | 313.20 | 2023-10-20 12:15:00 | 310.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-10-25 12:30:00 | 306.40 | 2023-10-27 09:15:00 | 310.35 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-10-26 09:15:00 | 307.05 | 2023-10-27 09:15:00 | 310.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-10-31 13:45:00 | 313.45 | 2023-11-01 11:15:00 | 308.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-11-02 12:45:00 | 309.95 | 2023-11-03 11:15:00 | 312.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-11-08 09:15:00 | 316.10 | 2023-11-13 12:15:00 | 347.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-30 14:45:00 | 342.35 | 2023-12-07 11:15:00 | 349.80 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2023-12-27 14:15:00 | 365.30 | 2024-01-05 14:15:00 | 384.30 | STOP_HIT | 1.00 | 5.20% |
| BUY | retest2 | 2023-12-27 15:00:00 | 365.60 | 2024-01-05 14:15:00 | 384.30 | STOP_HIT | 1.00 | 5.11% |
| BUY | retest2 | 2024-01-31 12:15:00 | 407.75 | 2024-02-05 13:15:00 | 448.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 13:45:00 | 408.20 | 2024-02-05 13:15:00 | 449.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-01 09:15:00 | 408.20 | 2024-02-05 13:15:00 | 449.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-02 09:15:00 | 412.30 | 2024-02-07 09:15:00 | 453.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-09 15:00:00 | 457.70 | 2024-02-12 09:15:00 | 437.35 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2024-02-23 10:45:00 | 444.45 | 2024-03-01 09:15:00 | 443.55 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-02-23 13:15:00 | 444.20 | 2024-03-01 09:15:00 | 443.55 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-02-23 15:00:00 | 443.70 | 2024-03-01 09:15:00 | 443.55 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-02-26 09:45:00 | 443.25 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-02-26 12:30:00 | 442.50 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-02-26 15:00:00 | 441.95 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-02-27 09:45:00 | 442.45 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-02-27 11:30:00 | 442.50 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-02-28 12:15:00 | 431.65 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-02-29 10:15:00 | 432.95 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-02-29 11:15:00 | 432.15 | 2024-03-01 10:15:00 | 443.85 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-03-06 12:30:00 | 455.85 | 2024-03-11 14:15:00 | 453.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-03-19 10:15:00 | 417.75 | 2024-03-21 09:15:00 | 429.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-03-19 15:00:00 | 418.55 | 2024-03-21 09:15:00 | 429.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-03-20 09:45:00 | 418.10 | 2024-03-21 09:15:00 | 429.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-03-20 11:00:00 | 419.15 | 2024-03-21 09:15:00 | 429.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-03-27 11:45:00 | 437.40 | 2024-03-27 14:15:00 | 430.55 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-03-28 14:00:00 | 437.60 | 2024-04-09 10:15:00 | 440.65 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2024-04-01 10:00:00 | 439.50 | 2024-04-09 10:15:00 | 440.65 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-04-16 09:30:00 | 455.25 | 2024-04-18 13:15:00 | 447.35 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-04-16 11:30:00 | 452.90 | 2024-04-18 13:15:00 | 447.35 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-04-16 12:45:00 | 453.30 | 2024-04-18 13:15:00 | 447.35 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-04-16 13:45:00 | 452.80 | 2024-04-18 13:15:00 | 447.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-04-23 10:30:00 | 441.70 | 2024-04-24 11:15:00 | 446.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-04-23 13:00:00 | 440.60 | 2024-04-24 11:15:00 | 446.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-04-23 14:15:00 | 440.20 | 2024-04-24 11:15:00 | 446.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-04-24 10:00:00 | 442.10 | 2024-04-24 11:15:00 | 446.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-04-25 11:00:00 | 449.00 | 2024-05-07 12:15:00 | 454.05 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2024-05-13 10:30:00 | 435.80 | 2024-05-15 09:15:00 | 459.10 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2024-05-17 09:15:00 | 472.20 | 2024-05-28 09:15:00 | 491.70 | STOP_HIT | 1.00 | 4.13% |
| BUY | retest2 | 2024-05-17 11:00:00 | 469.30 | 2024-05-28 09:15:00 | 491.70 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2024-05-17 13:00:00 | 470.00 | 2024-05-28 09:15:00 | 491.70 | STOP_HIT | 1.00 | 4.62% |
| SELL | retest2 | 2024-06-06 11:30:00 | 473.80 | 2024-06-06 14:15:00 | 472.50 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-06-12 09:15:00 | 486.45 | 2024-06-19 10:15:00 | 480.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-15 11:15:00 | 501.60 | 2024-07-15 12:15:00 | 497.05 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-07-16 09:15:00 | 503.15 | 2024-07-19 09:15:00 | 494.10 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-23 12:15:00 | 470.20 | 2024-07-24 10:15:00 | 498.30 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2024-07-23 15:15:00 | 485.70 | 2024-07-24 10:15:00 | 498.30 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-07-31 12:00:00 | 524.25 | 2024-08-05 09:15:00 | 513.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-08-01 09:15:00 | 537.20 | 2024-08-05 09:15:00 | 513.40 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-08-06 12:30:00 | 508.00 | 2024-08-07 09:15:00 | 519.45 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-08-06 13:00:00 | 507.40 | 2024-08-07 09:15:00 | 519.45 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-08-06 13:30:00 | 505.95 | 2024-08-07 09:15:00 | 519.45 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-06 14:00:00 | 504.55 | 2024-08-07 09:15:00 | 519.45 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-08-09 09:15:00 | 529.35 | 2024-08-13 13:15:00 | 521.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-08-09 13:15:00 | 528.40 | 2024-08-13 13:15:00 | 521.55 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-12 12:30:00 | 524.85 | 2024-08-13 13:15:00 | 521.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-08-12 13:45:00 | 524.55 | 2024-08-13 14:15:00 | 520.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-08-13 09:15:00 | 525.35 | 2024-08-13 14:15:00 | 520.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-13 10:00:00 | 524.50 | 2024-08-13 14:15:00 | 520.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-13 10:30:00 | 525.20 | 2024-08-13 14:15:00 | 520.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-23 09:30:00 | 532.40 | 2024-08-27 15:15:00 | 531.75 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-08-30 10:45:00 | 528.15 | 2024-09-04 10:15:00 | 501.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 12:00:00 | 528.05 | 2024-09-04 10:15:00 | 501.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 13:45:00 | 528.25 | 2024-09-04 10:15:00 | 501.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 15:00:00 | 524.50 | 2024-09-05 14:15:00 | 498.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 11:45:00 | 519.85 | 2024-09-06 09:15:00 | 493.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 520.70 | 2024-09-06 09:15:00 | 494.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 10:45:00 | 528.15 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 8.36% |
| SELL | retest2 | 2024-08-30 12:00:00 | 528.05 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 8.34% |
| SELL | retest2 | 2024-08-30 13:45:00 | 528.25 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 8.38% |
| SELL | retest2 | 2024-08-30 15:00:00 | 524.50 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 7.72% |
| SELL | retest2 | 2024-09-02 11:45:00 | 519.85 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 6.90% |
| SELL | retest2 | 2024-09-03 09:30:00 | 520.70 | 2024-09-09 14:15:00 | 484.00 | STOP_HIT | 0.50 | 7.05% |
| BUY | retest2 | 2024-09-30 11:45:00 | 511.25 | 2024-10-03 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-09-30 13:30:00 | 509.85 | 2024-10-03 09:15:00 | 505.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-30 14:45:00 | 510.85 | 2024-10-03 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-10-01 09:15:00 | 512.55 | 2024-10-03 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-22 10:15:00 | 481.30 | 2024-10-28 09:15:00 | 457.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 10:45:00 | 479.85 | 2024-10-28 09:15:00 | 455.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 481.30 | 2024-10-29 14:15:00 | 445.75 | STOP_HIT | 0.50 | 7.39% |
| SELL | retest2 | 2024-10-24 10:45:00 | 479.85 | 2024-10-29 14:15:00 | 445.75 | STOP_HIT | 0.50 | 7.11% |
| SELL | retest2 | 2024-11-08 09:15:00 | 429.20 | 2024-11-13 09:15:00 | 407.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 429.20 | 2024-11-14 11:15:00 | 411.85 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2024-12-05 14:45:00 | 415.50 | 2024-12-06 10:15:00 | 420.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-16 09:45:00 | 408.15 | 2024-12-16 13:15:00 | 411.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-17 09:15:00 | 408.55 | 2024-12-19 09:15:00 | 388.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 408.55 | 2024-12-23 10:15:00 | 387.50 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2025-01-02 13:15:00 | 388.30 | 2025-01-06 10:15:00 | 382.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-01-06 10:00:00 | 388.55 | 2025-01-06 10:15:00 | 382.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-01-09 09:15:00 | 376.25 | 2025-01-15 09:15:00 | 378.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-16 09:15:00 | 376.15 | 2025-01-21 12:15:00 | 383.30 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2025-01-21 11:30:00 | 384.90 | 2025-01-21 14:15:00 | 381.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-31 10:15:00 | 382.00 | 2025-02-03 09:15:00 | 370.70 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-02-14 10:30:00 | 357.50 | 2025-02-18 15:15:00 | 360.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-02-17 13:00:00 | 358.40 | 2025-02-18 15:15:00 | 360.20 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-02-18 09:15:00 | 355.50 | 2025-02-18 15:15:00 | 360.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-18 11:00:00 | 357.25 | 2025-02-18 15:15:00 | 360.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-02-20 11:00:00 | 364.35 | 2025-02-25 09:15:00 | 359.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-02-24 11:30:00 | 364.80 | 2025-02-25 09:15:00 | 359.95 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-17 09:15:00 | 385.60 | 2025-03-25 13:15:00 | 398.85 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2025-04-03 11:30:00 | 395.20 | 2025-04-03 13:15:00 | 397.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-04-09 09:45:00 | 376.10 | 2025-04-11 09:15:00 | 388.25 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-04-09 13:30:00 | 375.95 | 2025-04-11 09:15:00 | 388.25 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-04-17 11:15:00 | 394.10 | 2025-04-23 12:15:00 | 398.45 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2025-05-02 11:15:00 | 383.20 | 2025-05-08 09:15:00 | 388.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-05-02 14:15:00 | 383.35 | 2025-05-08 09:15:00 | 388.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-05-02 15:15:00 | 383.35 | 2025-05-08 09:15:00 | 388.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-05-06 09:45:00 | 381.15 | 2025-05-08 09:15:00 | 388.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-05-20 09:15:00 | 405.00 | 2025-05-21 15:15:00 | 405.40 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-05-29 12:30:00 | 397.80 | 2025-06-02 09:15:00 | 400.45 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-30 10:15:00 | 397.90 | 2025-06-02 13:15:00 | 400.10 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-05-30 12:30:00 | 398.35 | 2025-06-02 13:15:00 | 400.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-05-30 14:15:00 | 398.60 | 2025-06-02 13:15:00 | 400.10 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-06-02 09:15:00 | 396.70 | 2025-06-02 13:15:00 | 400.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-05 13:15:00 | 394.55 | 2025-06-06 09:15:00 | 398.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-17 12:15:00 | 392.60 | 2025-06-23 12:15:00 | 391.30 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-06-18 09:30:00 | 391.80 | 2025-06-23 12:15:00 | 391.30 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-06-26 15:15:00 | 394.00 | 2025-06-30 13:15:00 | 391.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-27 13:15:00 | 394.00 | 2025-06-30 13:15:00 | 391.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-27 14:00:00 | 394.00 | 2025-06-30 13:15:00 | 391.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-27 15:00:00 | 394.55 | 2025-06-30 13:15:00 | 391.05 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-14 11:15:00 | 382.00 | 2025-07-15 09:15:00 | 384.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-14 12:45:00 | 381.85 | 2025-07-15 09:15:00 | 384.70 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-14 13:30:00 | 382.65 | 2025-07-15 09:15:00 | 384.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-16 11:45:00 | 387.20 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-07-17 09:45:00 | 387.35 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-17 10:30:00 | 387.15 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-17 13:30:00 | 387.05 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-18 09:15:00 | 387.50 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-18 13:00:00 | 387.70 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-21 11:45:00 | 387.00 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 13:45:00 | 386.60 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-22 12:45:00 | 391.10 | 2025-07-24 11:15:00 | 384.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-01 11:30:00 | 372.20 | 2025-08-05 13:15:00 | 378.05 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-08-01 15:00:00 | 372.50 | 2025-08-05 13:15:00 | 378.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-08-06 11:00:00 | 377.00 | 2025-08-07 12:15:00 | 375.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-08-06 12:30:00 | 377.00 | 2025-08-07 12:15:00 | 375.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-08-06 14:45:00 | 376.85 | 2025-08-07 12:15:00 | 375.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-08-07 09:45:00 | 376.80 | 2025-08-07 12:15:00 | 375.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-11 12:15:00 | 381.00 | 2025-08-20 11:15:00 | 384.85 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-08-29 12:30:00 | 375.90 | 2025-09-01 12:15:00 | 376.05 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-29 13:00:00 | 376.00 | 2025-09-01 12:15:00 | 376.05 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-08-29 13:45:00 | 375.80 | 2025-09-01 12:15:00 | 376.05 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-08-29 15:00:00 | 375.30 | 2025-09-01 12:15:00 | 376.05 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-15 09:15:00 | 396.30 | 2025-09-16 13:15:00 | 394.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-09-15 15:15:00 | 394.95 | 2025-09-16 13:15:00 | 394.30 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-17 09:15:00 | 399.10 | 2025-09-18 12:15:00 | 391.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 392.45 | 2025-10-01 09:15:00 | 392.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-09-25 14:15:00 | 392.40 | 2025-10-01 11:15:00 | 391.30 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-09-25 15:00:00 | 392.45 | 2025-10-01 11:15:00 | 391.30 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-26 09:15:00 | 390.95 | 2025-10-01 11:15:00 | 391.30 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-09-30 13:45:00 | 388.30 | 2025-10-01 11:15:00 | 391.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-08 12:00:00 | 383.85 | 2025-10-09 10:15:00 | 385.85 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-10-09 12:45:00 | 384.05 | 2025-10-10 09:15:00 | 385.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-10-20 09:15:00 | 390.05 | 2025-10-28 14:15:00 | 390.85 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-10-20 10:00:00 | 388.95 | 2025-10-28 14:15:00 | 390.85 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-11-28 12:00:00 | 377.65 | 2025-11-28 13:15:00 | 375.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-01 14:00:00 | 378.85 | 2025-12-03 09:15:00 | 373.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-02 11:30:00 | 378.25 | 2025-12-03 09:15:00 | 373.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-08 11:15:00 | 378.15 | 2025-12-08 13:15:00 | 376.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-12-19 14:30:00 | 385.30 | 2025-12-30 10:15:00 | 397.25 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2025-12-19 15:00:00 | 385.80 | 2025-12-30 10:15:00 | 397.25 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest2 | 2025-12-22 09:15:00 | 386.35 | 2025-12-30 10:15:00 | 397.25 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-12-22 10:45:00 | 386.10 | 2025-12-30 10:15:00 | 397.25 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-12-22 14:15:00 | 385.95 | 2025-12-30 10:15:00 | 397.25 | STOP_HIT | 1.00 | 2.93% |
| SELL | retest2 | 2026-01-22 11:15:00 | 421.65 | 2026-01-22 15:15:00 | 423.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-01-30 15:00:00 | 440.80 | 2026-02-01 09:15:00 | 436.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-19 10:15:00 | 415.40 | 2026-02-20 09:15:00 | 424.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-19 12:00:00 | 415.60 | 2026-02-20 09:15:00 | 424.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-19 12:30:00 | 415.70 | 2026-02-20 09:15:00 | 424.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 415.60 | 2026-02-20 09:15:00 | 424.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-02-26 13:15:00 | 432.90 | 2026-02-27 09:15:00 | 427.65 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-26 14:00:00 | 432.85 | 2026-02-27 09:15:00 | 427.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-26 14:30:00 | 433.00 | 2026-02-27 09:15:00 | 427.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-26 15:00:00 | 434.15 | 2026-02-27 09:15:00 | 427.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-10 12:30:00 | 449.05 | 2026-04-17 15:15:00 | 438.60 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2026-04-23 11:45:00 | 448.75 | 2026-05-05 12:15:00 | 472.80 | STOP_HIT | 1.00 | 5.36% |
