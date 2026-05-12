# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 170.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 19 |
| ALERT3 | 126 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 44 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 30
- **Target hits / Stop hits / Partials:** 2 / 44 / 11
- **Avg / median % per leg:** 1.33% / -0.72%
- **Sum % (uncompounded):** 75.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 3 | 12.5% | 1 | 23 | 0 | -1.11% | -26.6% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.79% | 0.8% |
| BUY @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 1 | 22 | 0 | -1.19% | -27.4% |
| SELL (all) | 33 | 24 | 72.7% | 1 | 21 | 11 | 3.10% | 102.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.47% | 6.9% |
| SELL @ 3rd Alert (retest2) | 31 | 22 | 71.0% | 1 | 20 | 10 | 3.08% | 95.4% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 2.57% | 7.7% |
| retest2 (combined) | 54 | 24 | 44.4% | 2 | 42 | 10 | 1.26% | 68.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 216.10 | 211.54 | 210.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 218.76 | 213.92 | 212.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 225.91 | 228.06 | 225.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 229.63 | 227.91 | 225.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 233.20 | 225.58 | 224.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 231.51 | 233.49 | 233.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 231.51 | 233.49 | 233.49 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 234.10 | 233.39 | 233.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 235.15 | 234.29 | 233.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 234.40 | 234.82 | 234.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 234.40 | 234.82 | 234.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 234.06 | 234.67 | 234.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 235.71 | 234.63 | 234.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 233.15 | 236.14 | 236.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 233.15 | 236.14 | 236.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 232.15 | 235.34 | 235.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 234.75 | 233.32 | 234.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 234.75 | 233.32 | 234.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 233.31 | 233.32 | 234.27 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 235.83 | 234.72 | 234.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 236.70 | 235.11 | 234.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 234.88 | 235.25 | 234.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 234.88 | 235.25 | 234.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 234.99 | 235.20 | 234.99 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 231.98 | 234.35 | 234.63 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 237.35 | 234.80 | 234.75 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 233.85 | 234.76 | 234.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 233.32 | 234.47 | 234.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 233.58 | 233.57 | 234.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 233.58 | 233.57 | 234.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 234.37 | 233.73 | 234.13 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 238.39 | 235.05 | 234.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 240.35 | 237.26 | 235.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 239.49 | 239.54 | 237.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 240.00 | 239.73 | 238.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 239.10 | 239.78 | 238.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 239.10 | 239.78 | 238.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 238.72 | 239.57 | 238.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 238.07 | 239.57 | 238.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 240.02 | 239.66 | 238.90 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 237.80 | 238.50 | 238.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 236.92 | 238.18 | 238.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 227.95 | 227.30 | 230.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 227.41 | 227.30 | 230.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 229.28 | 228.53 | 229.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 230.21 | 228.53 | 229.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 228.92 | 228.61 | 229.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 229.70 | 228.61 | 229.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 222.00 | 221.26 | 222.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 220.60 | 221.26 | 222.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 220.62 | 221.08 | 222.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 225.95 | 222.54 | 222.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 225.95 | 222.54 | 222.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 227.66 | 224.73 | 223.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 224.86 | 224.95 | 223.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 227.20 | 224.95 | 223.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 231.00 | 230.34 | 228.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 231.21 | 230.41 | 229.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 231.15 | 230.38 | 229.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 231.20 | 230.50 | 229.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 231.40 | 230.50 | 229.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 229.00 | 231.44 | 230.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 229.00 | 231.44 | 230.72 | SL hit (close<ema400) qty=1.00 sl=230.72 alert=retest1 |

### Cycle 12 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 227.08 | 229.81 | 230.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 225.76 | 229.00 | 229.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 225.96 | 225.87 | 227.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 226.28 | 225.87 | 227.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 223.34 | 223.31 | 224.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 223.54 | 223.31 | 224.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 224.40 | 223.37 | 223.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 224.40 | 223.37 | 223.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 224.10 | 223.52 | 223.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 224.20 | 223.52 | 223.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 224.92 | 223.80 | 224.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 224.92 | 223.80 | 224.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 225.45 | 224.13 | 224.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 225.69 | 224.13 | 224.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 225.99 | 224.50 | 224.37 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 222.60 | 224.35 | 224.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 221.30 | 223.31 | 223.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 222.62 | 222.34 | 223.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 222.62 | 222.34 | 223.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 225.32 | 222.70 | 223.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 225.75 | 222.70 | 223.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 223.30 | 223.16 | 223.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 222.82 | 223.28 | 223.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 225.15 | 223.66 | 223.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 225.15 | 223.66 | 223.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 227.83 | 225.00 | 224.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 228.90 | 229.54 | 228.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:00:00 | 228.90 | 229.54 | 228.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 229.18 | 229.47 | 228.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:45:00 | 228.92 | 229.47 | 228.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 229.71 | 229.40 | 228.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 228.49 | 229.40 | 228.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 227.45 | 229.01 | 228.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 227.45 | 229.01 | 228.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 226.87 | 228.58 | 228.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 227.12 | 228.58 | 228.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 226.01 | 228.07 | 228.10 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 229.20 | 228.15 | 228.08 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 227.49 | 228.10 | 228.11 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 228.60 | 228.20 | 228.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 229.10 | 228.38 | 228.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 227.85 | 228.29 | 228.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 227.76 | 228.29 | 228.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 228.12 | 228.26 | 228.22 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 227.71 | 228.12 | 228.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 224.32 | 227.21 | 227.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 13:15:00 | 224.47 | 224.29 | 225.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 224.47 | 224.29 | 225.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 221.00 | 220.59 | 222.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 219.44 | 220.36 | 221.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 222.80 | 221.14 | 221.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 222.80 | 221.14 | 221.01 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 219.77 | 221.41 | 221.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 217.40 | 220.15 | 220.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 217.27 | 212.69 | 214.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 218.97 | 212.69 | 214.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 216.55 | 213.46 | 214.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 214.49 | 214.08 | 214.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 215.70 | 214.34 | 214.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 214.77 | 214.66 | 214.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 214.77 | 214.66 | 214.65 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 214.20 | 214.57 | 214.61 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 221.41 | 215.66 | 215.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 224.14 | 217.35 | 215.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 221.64 | 221.77 | 219.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 221.64 | 221.77 | 219.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 222.46 | 222.72 | 221.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 222.10 | 222.72 | 221.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 221.78 | 222.91 | 222.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 221.78 | 222.91 | 222.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 222.49 | 222.83 | 222.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 223.03 | 222.65 | 222.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 218.95 | 221.85 | 221.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 218.95 | 221.85 | 221.99 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 222.30 | 220.76 | 220.60 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 219.95 | 220.70 | 220.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 218.51 | 220.04 | 220.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 218.97 | 218.04 | 218.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 218.97 | 218.04 | 218.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 218.42 | 218.12 | 218.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 218.76 | 218.12 | 218.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 219.01 | 218.30 | 218.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 219.01 | 218.30 | 218.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 220.06 | 218.65 | 218.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 219.95 | 218.65 | 218.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 220.10 | 218.94 | 219.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 220.63 | 218.94 | 219.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 218.40 | 218.84 | 219.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 215.77 | 218.84 | 219.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 14:15:00 | 204.98 | 207.28 | 210.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 209.76 | 207.31 | 209.63 | SL hit (close>ema200) qty=0.50 sl=207.31 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 213.36 | 210.29 | 210.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 209.20 | 211.20 | 211.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 208.65 | 210.69 | 211.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 207.27 | 207.24 | 208.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 207.41 | 207.24 | 208.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 209.38 | 207.66 | 208.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 209.38 | 207.66 | 208.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 206.96 | 207.52 | 208.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 205.78 | 207.51 | 208.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 205.57 | 206.96 | 207.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 205.81 | 206.56 | 207.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 208.49 | 207.45 | 207.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 208.49 | 207.45 | 207.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 211.00 | 209.06 | 208.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 214.98 | 215.53 | 213.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 214.98 | 215.53 | 213.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 214.63 | 215.43 | 214.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 214.63 | 215.43 | 214.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 215.46 | 215.43 | 214.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 215.78 | 215.50 | 214.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 216.18 | 215.71 | 215.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 214.22 | 215.29 | 215.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 214.22 | 215.29 | 215.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 213.95 | 215.02 | 215.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 206.00 | 204.51 | 206.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 206.00 | 204.51 | 206.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 207.73 | 205.16 | 206.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 207.73 | 205.16 | 206.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 204.90 | 205.10 | 206.44 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 207.61 | 206.76 | 206.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 208.25 | 207.33 | 207.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 209.57 | 209.66 | 209.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:30:00 | 210.21 | 209.66 | 209.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 211.28 | 210.33 | 209.61 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 207.54 | 209.55 | 209.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 206.75 | 208.99 | 209.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 208.00 | 207.87 | 208.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 208.00 | 207.87 | 208.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 208.31 | 207.96 | 208.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 208.31 | 207.96 | 208.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 208.32 | 208.03 | 208.56 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 211.32 | 209.28 | 209.03 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 207.80 | 209.27 | 209.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 206.73 | 208.31 | 208.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 205.74 | 205.32 | 206.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 205.74 | 205.32 | 206.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 206.65 | 205.76 | 206.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 206.40 | 205.76 | 206.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 209.31 | 206.47 | 206.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 209.31 | 206.47 | 206.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 211.65 | 207.51 | 207.36 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 207.31 | 208.62 | 208.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 205.93 | 208.08 | 208.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 207.73 | 207.30 | 207.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 207.80 | 207.30 | 207.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 207.60 | 207.36 | 207.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:30:00 | 207.91 | 207.36 | 207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 208.67 | 207.62 | 207.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 208.67 | 207.62 | 207.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 209.03 | 207.90 | 207.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 209.03 | 207.90 | 207.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 208.75 | 208.07 | 208.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 209.39 | 208.49 | 208.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 207.70 | 208.33 | 208.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 207.70 | 208.33 | 208.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 210.65 | 209.91 | 209.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 213.55 | 209.56 | 209.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 211.41 | 212.23 | 211.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 211.28 | 212.02 | 211.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 211.30 | 211.84 | 211.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 210.49 | 211.57 | 211.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 210.49 | 211.57 | 211.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 210.66 | 211.39 | 211.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 210.00 | 211.39 | 211.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 210.99 | 211.31 | 211.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 211.69 | 211.31 | 211.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 208.54 | 212.58 | 213.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 208.54 | 212.58 | 213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 207.23 | 211.51 | 212.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 188.96 | 188.62 | 192.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:45:00 | 189.33 | 188.62 | 192.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 185.30 | 184.46 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 184.35 | 184.46 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 184.67 | 184.50 | 185.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 183.80 | 184.50 | 185.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 184.05 | 184.41 | 185.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 183.99 | 184.34 | 185.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 184.03 | 184.32 | 185.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 182.55 | 183.93 | 184.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 182.22 | 183.93 | 184.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 182.44 | 183.49 | 184.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:30:00 | 182.52 | 182.86 | 183.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.61 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.85 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.79 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 174.83 | 176.67 | 178.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.11 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.32 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 173.39 | 175.47 | 177.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 173.41 | 173.01 | 174.58 | SL hit (close>ema200) qty=0.50 sl=173.01 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 175.80 | 175.13 | 175.05 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 174.70 | 175.01 | 175.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 172.93 | 174.42 | 174.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 171.65 | 171.63 | 172.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:30:00 | 171.68 | 171.63 | 172.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 172.56 | 171.81 | 172.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 172.65 | 171.81 | 172.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 172.40 | 171.93 | 172.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:15:00 | 172.47 | 171.93 | 172.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 172.53 | 172.05 | 172.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 172.43 | 172.05 | 172.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 172.24 | 172.09 | 172.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 171.15 | 172.14 | 172.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 162.59 | 165.25 | 167.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 163.41 | 163.21 | 164.98 | SL hit (close>ema200) qty=0.50 sl=163.21 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 156.88 | 155.79 | 155.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 158.35 | 156.71 | 156.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 160.79 | 161.13 | 160.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 160.79 | 161.13 | 160.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 160.30 | 160.97 | 160.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 160.30 | 160.97 | 160.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 159.99 | 160.67 | 160.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 159.99 | 160.67 | 160.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 157.96 | 160.13 | 159.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 157.96 | 160.13 | 159.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 158.71 | 159.84 | 159.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 157.88 | 159.84 | 159.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 160.74 | 160.43 | 160.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 160.56 | 160.43 | 160.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 162.60 | 160.87 | 160.33 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 156.81 | 159.84 | 160.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 153.73 | 158.08 | 159.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 161.90 | 157.49 | 158.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 161.90 | 157.49 | 158.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 163.05 | 158.60 | 158.71 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 162.50 | 159.38 | 159.05 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 160.00 | 160.57 | 160.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 159.27 | 160.31 | 160.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 157.75 | 157.74 | 158.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 157.75 | 157.74 | 158.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 156.68 | 157.50 | 158.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 155.91 | 157.04 | 157.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 148.11 | 150.50 | 153.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 149.67 | 148.55 | 150.47 | SL hit (close>ema200) qty=0.50 sl=148.55 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 151.65 | 150.31 | 150.17 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 149.25 | 150.19 | 150.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 148.07 | 149.51 | 149.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 143.70 | 143.54 | 145.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 143.70 | 143.54 | 145.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 148.23 | 143.88 | 144.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 148.30 | 143.88 | 144.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 146.50 | 145.49 | 145.37 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 143.86 | 145.02 | 145.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 142.92 | 144.36 | 144.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 142.01 | 141.34 | 142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 144.31 | 141.34 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 143.20 | 141.71 | 142.58 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 146.64 | 143.41 | 143.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 147.76 | 145.16 | 144.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 145.77 | 145.91 | 144.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 145.77 | 145.91 | 144.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 144.80 | 145.69 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 144.87 | 145.69 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 144.69 | 145.49 | 144.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 144.69 | 145.49 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 145.49 | 145.49 | 144.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 145.63 | 145.40 | 144.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 145.66 | 145.29 | 144.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 146.36 | 145.37 | 144.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 146.28 | 145.62 | 145.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 147.54 | 147.49 | 146.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 147.60 | 147.49 | 146.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 146.48 | 147.29 | 146.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 146.48 | 147.29 | 146.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 145.84 | 147.00 | 146.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 145.84 | 147.00 | 146.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 146.01 | 146.80 | 146.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 145.39 | 146.80 | 146.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 146.23 | 146.69 | 146.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 144.04 | 146.16 | 146.14 | SL hit (close<static) qty=1.00 sl=144.55 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 144.62 | 145.85 | 146.00 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 147.94 | 146.19 | 146.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 150.98 | 147.44 | 146.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 153.59 | 154.46 | 152.47 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 146.37 | 151.57 | 151.86 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 156.80 | 152.46 | 152.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 157.95 | 154.65 | 153.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 157.94 | 158.61 | 156.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 157.94 | 158.61 | 156.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 157.90 | 158.47 | 157.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 155.94 | 158.47 | 157.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 155.87 | 157.95 | 156.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 154.77 | 157.95 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 155.84 | 157.53 | 156.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 155.84 | 157.53 | 156.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 156.05 | 156.45 | 156.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 154.46 | 155.92 | 156.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 155.22 | 155.07 | 155.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 155.22 | 155.07 | 155.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 152.19 | 150.89 | 152.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 152.19 | 150.89 | 152.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 151.37 | 150.99 | 151.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:00:00 | 151.08 | 151.55 | 151.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-19 09:15:00 | 135.97 | 149.74 | 150.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 150.18 | 149.55 | 149.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 13:15:00 | 151.46 | 150.16 | 149.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 152.29 | 153.09 | 152.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 152.29 | 153.09 | 152.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 152.55 | 152.98 | 152.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 147.19 | 152.98 | 152.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 149.14 | 152.21 | 152.03 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 149.36 | 151.64 | 151.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 148.32 | 150.98 | 151.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 150.29 | 149.94 | 150.72 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 145.40 | 149.94 | 150.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 144.85 | 143.67 | 145.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 145.98 | 143.67 | 145.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 138.13 | 141.39 | 143.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 142.59 | 140.79 | 141.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 142.59 | 140.79 | 141.83 | SL hit (close>ema200) qty=0.50 sl=140.79 alert=retest1 |

### Cycle 59 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 144.46 | 142.42 | 142.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 145.10 | 143.63 | 143.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 146.56 | 147.90 | 146.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 149.38 | 148.19 | 146.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 149.38 | 148.19 | 146.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 149.85 | 148.52 | 146.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:30:00 | 150.71 | 149.02 | 147.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 149.67 | 149.10 | 147.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 144.10 | 147.38 | 147.27 | SL hit (close<static) qty=1.00 sl=146.50 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 145.19 | 146.94 | 147.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 141.90 | 145.18 | 146.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 142.90 | 142.33 | 143.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:15:00 | 142.42 | 142.33 | 143.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 140.67 | 142.00 | 143.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 140.26 | 142.00 | 143.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 143.78 | 142.18 | 143.20 | SL hit (close>static) qty=1.00 sl=143.54 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 146.37 | 143.86 | 143.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 147.04 | 145.24 | 144.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 142.70 | 144.73 | 144.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 142.79 | 144.73 | 144.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 142.00 | 144.19 | 144.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 142.00 | 144.19 | 144.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 141.98 | 143.75 | 143.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 141.40 | 143.28 | 143.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 135.54 | 134.64 | 136.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 135.54 | 134.64 | 136.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 136.75 | 135.06 | 136.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 136.80 | 135.06 | 136.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 136.10 | 135.27 | 136.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 137.15 | 135.27 | 136.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 141.65 | 136.73 | 137.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 141.29 | 136.73 | 137.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 142.10 | 137.80 | 137.67 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 134.90 | 137.86 | 138.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 134.66 | 137.22 | 137.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 137.20 | 136.87 | 137.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 137.20 | 136.87 | 137.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 136.49 | 136.80 | 137.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 136.49 | 136.80 | 137.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 135.50 | 136.35 | 137.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:45:00 | 136.64 | 136.35 | 137.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 138.67 | 134.73 | 135.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 139.21 | 134.73 | 135.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 137.33 | 135.25 | 135.83 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 139.55 | 136.75 | 136.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 140.25 | 137.86 | 137.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 139.87 | 140.07 | 138.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 139.87 | 140.07 | 138.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 140.20 | 141.05 | 139.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 139.86 | 141.05 | 139.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 139.89 | 140.67 | 139.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:30:00 | 139.56 | 140.67 | 139.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 139.16 | 140.37 | 139.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 138.99 | 140.37 | 139.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 138.70 | 140.03 | 139.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 138.65 | 140.03 | 139.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 140.58 | 140.18 | 139.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 144.44 | 140.18 | 139.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 14:15:00 | 158.88 | 157.11 | 154.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 162.00 | 162.57 | 162.61 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 164.20 | 162.90 | 162.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 164.55 | 163.86 | 163.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 163.30 | 164.34 | 163.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 163.30 | 164.34 | 163.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 165.10 | 164.49 | 163.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 162.87 | 164.49 | 163.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 164.21 | 165.25 | 164.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 164.21 | 165.25 | 164.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 165.00 | 165.20 | 164.72 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 161.91 | 164.40 | 164.43 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 165.96 | 164.32 | 164.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 168.70 | 167.36 | 166.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 167.37 | 167.37 | 166.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:45:00 | 166.99 | 167.37 | 166.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 169.50 | 169.90 | 169.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 169.67 | 169.90 | 169.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 170.00 | 169.92 | 169.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 169.56 | 169.92 | 169.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 10:45:00 | 229.63 | 2025-05-23 09:15:00 | 231.51 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-05-19 09:15:00 | 233.20 | 2025-05-23 09:15:00 | 231.51 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-05-28 09:15:00 | 235.71 | 2025-05-30 09:15:00 | 233.15 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-20 12:15:00 | 220.60 | 2025-06-24 09:15:00 | 225.95 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-06-23 09:15:00 | 220.62 | 2025-06-24 09:15:00 | 225.95 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest1 | 2025-06-25 09:15:00 | 227.20 | 2025-07-02 09:15:00 | 229.00 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-06-30 09:30:00 | 231.21 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-01 10:00:00 | 231.15 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-01 11:30:00 | 231.20 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-01 12:00:00 | 231.40 | 2025-07-02 10:15:00 | 226.67 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-14 13:45:00 | 222.82 | 2025-07-14 14:15:00 | 225.15 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-28 12:00:00 | 219.44 | 2025-07-29 14:15:00 | 222.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-08-06 11:30:00 | 214.49 | 2025-08-06 15:15:00 | 214.77 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-08-06 12:30:00 | 215.70 | 2025-08-06 15:15:00 | 214.77 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-08-13 14:30:00 | 223.03 | 2025-08-14 09:15:00 | 218.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-26 09:15:00 | 215.77 | 2025-08-29 14:15:00 | 204.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 215.77 | 2025-09-01 09:15:00 | 209.76 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-09-09 10:15:00 | 205.78 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-09 12:00:00 | 205.57 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-09 14:30:00 | 205.81 | 2025-09-11 09:15:00 | 208.49 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-18 15:00:00 | 215.78 | 2025-09-22 13:15:00 | 214.22 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-19 10:00:00 | 216.18 | 2025-09-22 13:15:00 | 214.22 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-27 09:15:00 | 213.55 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-10-28 11:15:00 | 211.41 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-28 11:45:00 | 211.28 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-28 12:30:00 | 211.30 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-29 09:15:00 | 211.69 | 2025-11-04 10:15:00 | 208.54 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-17 11:15:00 | 183.80 | 2025-11-24 09:15:00 | 174.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 12:00:00 | 184.05 | 2025-11-24 09:15:00 | 174.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:00:00 | 183.99 | 2025-11-24 09:15:00 | 174.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 14:30:00 | 184.03 | 2025-11-24 09:15:00 | 174.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 10:15:00 | 182.22 | 2025-11-24 11:15:00 | 173.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 12:15:00 | 182.44 | 2025-11-24 11:15:00 | 173.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 14:30:00 | 182.52 | 2025-11-24 11:15:00 | 173.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 183.80 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2025-11-17 12:00:00 | 184.05 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2025-11-17 14:00:00 | 183.99 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.75% |
| SELL | retest2 | 2025-11-17 14:30:00 | 184.03 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-11-18 10:15:00 | 182.22 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-11-18 12:15:00 | 182.44 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-11-18 14:30:00 | 182.52 | 2025-11-25 14:15:00 | 173.41 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-12-03 09:15:00 | 171.15 | 2025-12-08 13:15:00 | 162.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 171.15 | 2025-12-09 13:15:00 | 163.41 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-01-08 10:45:00 | 155.91 | 2026-01-09 15:15:00 | 148.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 155.91 | 2026-01-12 15:15:00 | 149.67 | STOP_HIT | 0.50 | 4.00% |
| BUY | retest2 | 2026-01-29 13:30:00 | 145.63 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-30 10:00:00 | 145.66 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-01-30 10:30:00 | 146.36 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-30 15:00:00 | 146.28 | 2026-02-02 10:15:00 | 144.04 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-02-18 10:00:00 | 151.08 | 2026-02-19 09:15:00 | 135.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-03-04 09:15:00 | 145.40 | 2026-03-09 09:15:00 | 138.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-04 09:15:00 | 145.40 | 2026-03-10 09:15:00 | 142.59 | STOP_HIT | 0.50 | 1.93% |
| BUY | retest2 | 2026-03-12 12:00:00 | 149.85 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2026-03-12 13:30:00 | 150.71 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2026-03-12 14:30:00 | 149.67 | 2026-03-13 12:15:00 | 144.10 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-03-17 11:15:00 | 140.26 | 2026-03-17 13:15:00 | 143.78 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-08 09:15:00 | 144.44 | 2026-04-16 14:15:00 | 158.88 | TARGET_HIT | 1.00 | 10.00% |
