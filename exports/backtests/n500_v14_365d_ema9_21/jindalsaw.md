# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 243.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 26 |
| ALERT3 | 153 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 19 |
| TARGET_HIT | 1 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 43
- **Target hits / Stop hits / Partials:** 1 / 65 / 19
- **Avg / median % per leg:** 0.79% / 0.00%
- **Sum % (uncompounded):** 67.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 1 | 12 | 0 | -0.26% | -3.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.39% | -1.4% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 1 | 11 | 0 | -0.16% | -2.0% |
| SELL (all) | 72 | 40 | 55.6% | 0 | 53 | 19 | 0.98% | 70.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 40 | 55.6% | 0 | 53 | 19 | 0.98% | 70.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.39% | -1.4% |
| retest2 (combined) | 84 | 42 | 50.0% | 1 | 64 | 19 | 0.82% | 68.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 219.00 | 214.59 | 214.31 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 11:15:00 | 212.80 | 213.99 | 214.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 212.25 | 213.64 | 213.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 15:15:00 | 213.49 | 213.30 | 213.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 216.00 | 213.30 | 213.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 216.28 | 213.89 | 213.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 216.98 | 213.89 | 213.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

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
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 215.20 | 218.50 | 218.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-20 13:15:00)

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

### Cycle 5 — BUY (started 2025-05-23 14:15:00)

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
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 213.24 | 214.96 | 214.91 | SL hit (close<static) qty=1.00 sl=213.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 213.24 | 214.96 | 214.91 | SL hit (close<static) qty=1.00 sl=213.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 213.24 | 214.96 | 214.91 | SL hit (close<static) qty=1.00 sl=213.63 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 213.24 | 214.96 | 214.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-28 15:15:00)

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

### Cycle 7 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 214.44 | 213.19 | 213.17 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 09:15:00)

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

### Cycle 9 — BUY (started 2025-06-04 13:15:00)

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

### Cycle 10 — SELL (started 2025-06-13 12:15:00)

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
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 237.10 | 235.23 | 235.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 239.30 | 236.05 | 235.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 235.75 | 236.77 | 235.97 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 234.71 | 235.82 | 235.82 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-24 09:15:00)

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

### Cycle 14 — SELL (started 2025-07-01 12:15:00)

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
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 224.06 | 224.02 | 225.76 | SL hit (close>ema200) qty=0.50 sl=224.02 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 228.41 | 225.79 | 225.56 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 226.65 | 227.28 | 227.28 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 228.30 | 227.48 | 227.37 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-22 12:15:00)

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

### Cycle 19 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 229.05 | 227.35 | 227.13 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-25 10:15:00)

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

### Cycle 21 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 213.20 | 209.87 | 209.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 214.39 | 211.33 | 210.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 210.10 | 211.54 | 210.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 206.47 | 210.34 | 210.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-06 10:15:00)

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

### Cycle 23 — BUY (started 2025-08-19 09:15:00)

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
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 203.41 | 205.57 | 205.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 203.10 | 205.08 | 204.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 203.13 | 205.08 | 204.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 203.11 | 204.68 | 204.80 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 207.40 | 204.74 | 204.69 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 201.57 | 204.62 | 204.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 200.21 | 202.29 | 203.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 201.30 | 200.97 | 202.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 201.30 | 200.97 | 202.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 199.91 | 200.76 | 202.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 199.91 | 200.76 | 202.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 197.60 | 199.54 | 200.94 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-02 09:15:00)

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

### Cycle 28 — SELL (started 2025-09-05 10:15:00)

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
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 202.84 | 202.38 | 202.48 | SL hit (close>static) qty=1.00 sl=202.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 202.84 | 202.38 | 202.48 | SL hit (close>static) qty=1.00 sl=202.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 202.84 | 202.38 | 202.48 | SL hit (close>static) qty=1.00 sl=202.79 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 203.00 | 202.38 | 202.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 208.25 | 203.56 | 203.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 208.25 | 203.56 | 203.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 208.25 | 203.56 | 203.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 208.25 | 203.56 | 203.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-12 09:15:00)

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

### Cycle 30 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 208.69 | 209.05 | 209.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 207.58 | 208.75 | 208.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 209.77 | 208.73 | 208.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 209.77 | 208.73 | 208.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-19 13:15:00)

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

### Cycle 32 — SELL (started 2025-09-25 14:15:00)

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

### Cycle 33 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 206.69 | 205.03 | 204.83 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 204.00 | 204.73 | 204.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 202.38 | 203.94 | 204.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 202.74 | 202.31 | 202.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 203.00 | 202.31 | 202.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 201.54 | 202.16 | 202.81 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-09 09:15:00)

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

### Cycle 36 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 202.99 | 204.13 | 204.22 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 205.50 | 204.40 | 204.29 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-14 09:15:00)

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
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 191.16 | 188.97 | 192.59 | SL hit (close>ema200) qty=0.50 sl=188.97 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-10 11:15:00)

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

### Cycle 40 — SELL (started 2025-11-14 09:15:00)

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
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 167.14 | 163.84 | 163.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 167.68 | 164.61 | 164.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 164.36 | 166.16 | 165.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 164.66 | 166.16 | 165.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 164.50 | 165.83 | 165.58 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-28 12:15:00)

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

### Cycle 43 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 168.79 | 165.74 | 165.36 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 12:15:00)

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
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 157.70 | 157.54 | 159.43 | SL hit (close>ema200) qty=0.50 sl=157.54 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 157.70 | 157.54 | 159.43 | SL hit (close>ema200) qty=0.50 sl=157.54 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 157.70 | 157.54 | 159.43 | SL hit (close>ema200) qty=0.50 sl=157.54 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 158.13 | 157.81 | 159.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 158.13 | 157.81 | 159.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 158.86 | 158.02 | 159.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 162.90 | 158.02 | 159.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 162.97 | 159.01 | 159.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 162.89 | 159.01 | 159.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 162.52 | 159.71 | 159.81 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 162.30 | 160.23 | 160.04 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-11 10:15:00)

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

### Cycle 47 — BUY (started 2025-12-12 15:15:00)

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

### Cycle 48 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 161.20 | 161.87 | 161.93 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-18 09:15:00)

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

### Cycle 50 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 161.30 | 162.53 | 162.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 14:15:00 | 160.94 | 161.82 | 162.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 164.05 | 162.23 | 162.31 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 164.19 | 162.62 | 162.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 164.70 | 163.84 | 163.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 167.71 | 167.76 | 166.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 167.71 | 167.76 | 166.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 167.05 | 167.14 | 166.53 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 165.11 | 166.38 | 166.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 164.17 | 165.93 | 166.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 165.65 | 165.41 | 165.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 164.55 | 165.41 | 165.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 164.46 | 165.22 | 165.70 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-31 09:15:00)

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

### Cycle 54 — SELL (started 2026-01-07 11:15:00)

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
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 171.75 | 159.75 | 160.10 | SL hit (close>static) qty=1.00 sl=162.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 174.40 | 162.68 | 161.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 11:15:00 | 176.85 | 165.51 | 162.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 188.50 | 189.86 | 184.44 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-23 12:15:00)

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
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 181.68 | 175.78 | 175.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 181.68 | 175.78 | 175.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 181.68 | 175.78 | 175.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

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

### Cycle 58 — SELL (started 2026-02-13 09:15:00)

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
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 180.56 | 180.55 | 182.98 | SL hit (close>ema200) qty=0.50 sl=180.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 180.56 | 180.55 | 182.98 | SL hit (close>ema200) qty=0.50 sl=180.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 180.56 | 180.55 | 182.98 | SL hit (close>ema200) qty=0.50 sl=180.55 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 180.54 | 180.71 | 182.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 180.09 | 180.71 | 182.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 184.83 | 178.18 | 178.98 | SL hit (close>static) qty=1.00 sl=184.33 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 187.62 | 180.07 | 179.76 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-02 10:15:00)

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
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 164.06 | 163.83 | 166.29 | SL hit (close>ema200) qty=0.50 sl=163.83 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-11 09:15:00)

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
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 193.61 | 195.75 | 195.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 191.50 | 194.90 | 195.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 194.91 | 192.68 | 194.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 185.30 | 193.57 | 193.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 188.00 | 185.80 | 185.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 14:15:00)

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

### Cycle 64 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 183.50 | 186.18 | 186.36 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 187.75 | 186.50 | 186.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 10:15:00 | 190.48 | 187.29 | 186.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 186.62 | 189.29 | 188.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 190.79 | 188.89 | 188.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 209.87 | 206.65 | 204.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-28 13:15:00)

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
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 233.25 | 230.78 | 231.15 | SL hit (close>static) qty=1.00 sl=233.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 233.25 | 230.78 | 231.15 | SL hit (close>static) qty=1.00 sl=233.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 234.50 | 231.86 | 231.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 238.00 | 233.68 | 232.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 243.30 | 243.99 | 241.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 243.36 | 243.99 | 241.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
