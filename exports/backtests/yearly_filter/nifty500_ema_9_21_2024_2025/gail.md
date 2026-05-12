# GAIL (India) Ltd. (GAIL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 166.59
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 100 |
| ALERT2 | 98 |
| ALERT2_SKIP | 43 |
| ALERT3 | 243 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 124 |
| PARTIAL | 10 |
| TARGET_HIT | 0 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 98
- **Target hits / Stop hits / Partials:** 0 / 127 / 10
- **Avg / median % per leg:** -0.24% / -0.76%
- **Sum % (uncompounded):** -33.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 21 | 29.2% | 0 | 69 | 3 | -0.21% | -15.4% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.87% | 23.2% |
| BUY @ 3rd Alert (retest2) | 66 | 15 | 22.7% | 0 | 66 | 0 | -0.59% | -38.6% |
| SELL (all) | 65 | 18 | 27.7% | 0 | 58 | 7 | -0.28% | -18.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 65 | 18 | 27.7% | 0 | 58 | 7 | -0.28% | -18.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.87% | 23.2% |
| retest2 (combined) | 131 | 33 | 25.2% | 0 | 124 | 7 | -0.43% | -56.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 197.65 | 194.31 | 194.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 199.70 | 195.39 | 194.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 199.70 | 199.78 | 198.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 199.70 | 199.78 | 198.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 199.75 | 199.77 | 198.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 199.75 | 199.77 | 198.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 197.75 | 199.37 | 198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 197.75 | 199.37 | 198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 197.10 | 198.91 | 198.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 197.05 | 198.91 | 198.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 196.55 | 197.87 | 197.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 199.10 | 197.87 | 197.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 09:15:00 | 196.40 | 197.58 | 197.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 196.40 | 197.58 | 197.68 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 204.70 | 198.75 | 198.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 205.00 | 200.00 | 198.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 205.85 | 206.07 | 203.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 205.85 | 206.07 | 203.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 199.95 | 204.41 | 203.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 200.50 | 204.41 | 203.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 201.65 | 203.85 | 203.12 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 200.25 | 202.24 | 202.49 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 205.70 | 201.91 | 201.89 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 201.25 | 202.51 | 202.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 199.10 | 201.63 | 202.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 200.40 | 200.06 | 200.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:00:00 | 200.40 | 200.06 | 200.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 200.75 | 200.27 | 200.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 200.55 | 200.27 | 200.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 200.85 | 200.38 | 200.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 202.30 | 200.38 | 200.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 200.50 | 200.41 | 200.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 12:15:00 | 199.55 | 200.64 | 200.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 199.75 | 200.46 | 200.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 12:15:00 | 203.10 | 200.99 | 200.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 203.10 | 200.99 | 200.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 220.75 | 206.28 | 203.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 217.55 | 220.81 | 213.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 217.55 | 220.81 | 213.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 217.55 | 220.81 | 213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 210.15 | 220.81 | 213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 196.20 | 215.89 | 212.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 196.20 | 215.89 | 212.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 177.00 | 208.11 | 208.93 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 206.15 | 200.99 | 200.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 211.30 | 206.18 | 203.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 209.10 | 210.92 | 208.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 209.10 | 210.92 | 208.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 209.10 | 210.92 | 208.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 209.10 | 210.92 | 208.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 206.62 | 210.06 | 208.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 209.68 | 209.56 | 208.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 218.15 | 219.42 | 219.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 218.15 | 219.42 | 219.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 216.11 | 218.75 | 219.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 219.28 | 218.33 | 218.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 219.28 | 218.33 | 218.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 219.28 | 218.33 | 218.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 219.37 | 218.33 | 218.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 219.78 | 218.62 | 218.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:15:00 | 217.50 | 218.79 | 219.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 218.48 | 218.78 | 218.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 10:15:00 | 215.00 | 213.45 | 213.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 215.00 | 213.45 | 213.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 218.28 | 215.06 | 214.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 218.96 | 219.34 | 217.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 218.96 | 219.34 | 217.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 221.07 | 221.92 | 220.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 221.07 | 221.92 | 220.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 221.74 | 221.70 | 220.73 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 219.55 | 220.61 | 220.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 10:15:00 | 217.88 | 220.06 | 220.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 13:15:00 | 220.08 | 219.62 | 220.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 13:15:00 | 220.08 | 219.62 | 220.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 220.08 | 219.62 | 220.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 220.08 | 219.62 | 220.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 219.20 | 219.54 | 219.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:30:00 | 220.00 | 219.54 | 219.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 220.89 | 219.75 | 219.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:00:00 | 220.89 | 219.75 | 219.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 220.44 | 219.89 | 220.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 221.47 | 219.89 | 220.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 221.40 | 220.19 | 220.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 223.10 | 220.77 | 220.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 11:15:00 | 221.59 | 221.95 | 221.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 12:00:00 | 221.59 | 221.95 | 221.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 221.90 | 221.94 | 221.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:15:00 | 222.27 | 221.98 | 221.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 230.15 | 232.32 | 232.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 230.15 | 232.32 | 232.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 221.87 | 228.77 | 230.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 223.40 | 222.89 | 226.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 223.40 | 222.89 | 226.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 224.99 | 223.62 | 225.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 224.85 | 223.62 | 225.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 221.80 | 222.95 | 224.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 213.19 | 222.81 | 224.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 221.09 | 221.29 | 222.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:30:00 | 220.75 | 221.30 | 222.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 228.89 | 223.78 | 223.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 228.89 | 223.78 | 223.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 12:15:00 | 229.01 | 225.51 | 224.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 230.63 | 230.80 | 229.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 13:15:00 | 231.82 | 230.80 | 229.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 14:45:00 | 231.98 | 231.02 | 229.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 15:15:00 | 232.00 | 231.02 | 229.49 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:15:00 | 243.41 | 235.58 | 232.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:15:00 | 243.58 | 235.58 | 232.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:15:00 | 243.60 | 235.58 | 232.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 238.30 | 239.73 | 237.35 | SL hit (close<ema200) qty=0.50 sl=239.73 alert=retest1 |

### Cycle 16 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 229.06 | 235.99 | 236.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 223.20 | 233.43 | 235.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 228.63 | 227.67 | 231.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 228.63 | 227.67 | 231.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 229.03 | 226.95 | 229.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 229.34 | 226.95 | 229.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 229.94 | 227.55 | 229.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 230.77 | 227.55 | 229.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 229.04 | 227.85 | 229.18 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 233.50 | 230.54 | 230.14 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 229.07 | 229.88 | 229.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 228.70 | 229.64 | 229.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 229.14 | 228.84 | 229.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 229.14 | 228.84 | 229.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 229.14 | 228.84 | 229.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 228.33 | 228.84 | 229.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 228.42 | 228.37 | 228.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 231.11 | 229.35 | 229.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 231.11 | 229.35 | 229.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 232.30 | 229.94 | 229.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 231.24 | 231.45 | 230.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:00:00 | 231.24 | 231.45 | 230.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 230.00 | 231.16 | 230.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 230.10 | 231.16 | 230.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 227.96 | 230.52 | 230.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 227.96 | 230.52 | 230.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 226.68 | 229.75 | 229.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 226.46 | 228.05 | 228.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 228.28 | 227.91 | 228.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 228.28 | 227.91 | 228.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 228.28 | 227.91 | 228.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 228.28 | 227.91 | 228.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 230.12 | 228.35 | 228.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 230.12 | 228.35 | 228.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 231.13 | 228.91 | 228.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 231.13 | 228.91 | 228.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 231.72 | 229.47 | 229.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 232.43 | 230.06 | 229.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 237.17 | 237.64 | 235.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:45:00 | 237.27 | 237.64 | 235.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 237.00 | 237.21 | 236.26 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 234.40 | 235.82 | 235.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 234.00 | 235.25 | 235.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 231.83 | 231.73 | 233.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:30:00 | 230.78 | 231.73 | 233.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 233.49 | 232.11 | 233.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 233.49 | 232.11 | 233.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 234.45 | 232.58 | 233.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 234.45 | 232.58 | 233.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 234.70 | 233.00 | 233.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 234.89 | 233.00 | 233.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 235.36 | 233.82 | 233.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 236.04 | 234.27 | 233.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 235.39 | 235.94 | 235.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 235.39 | 235.94 | 235.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 235.39 | 235.94 | 235.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 235.39 | 235.94 | 235.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 235.88 | 235.93 | 235.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 11:30:00 | 236.50 | 236.07 | 235.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 234.80 | 235.72 | 235.42 | SL hit (close<static) qty=1.00 sl=235.10 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 231.60 | 234.56 | 234.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 230.57 | 232.97 | 234.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 235.41 | 233.23 | 233.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 235.41 | 233.23 | 233.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 235.41 | 233.23 | 233.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 235.80 | 233.23 | 233.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 234.06 | 233.40 | 233.89 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 237.08 | 234.56 | 234.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 237.45 | 235.14 | 234.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 235.00 | 235.62 | 235.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 235.00 | 235.62 | 235.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 235.00 | 235.62 | 235.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 235.00 | 235.62 | 235.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 234.70 | 235.44 | 235.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 234.70 | 235.44 | 235.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 234.10 | 235.17 | 234.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:45:00 | 234.28 | 235.17 | 234.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 233.90 | 234.92 | 234.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 233.60 | 234.92 | 234.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 234.48 | 234.78 | 234.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 10:15:00 | 233.00 | 234.43 | 234.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 219.92 | 219.00 | 221.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 219.92 | 219.00 | 221.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 221.12 | 219.70 | 221.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:45:00 | 221.13 | 219.70 | 221.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 221.41 | 220.04 | 221.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 221.41 | 220.04 | 221.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 220.09 | 220.05 | 221.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 219.08 | 219.81 | 221.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:00:00 | 218.82 | 219.61 | 220.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:00:00 | 218.40 | 218.19 | 219.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 220.75 | 219.68 | 219.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 220.75 | 219.68 | 219.57 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 218.78 | 219.45 | 219.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 217.84 | 219.03 | 219.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 218.99 | 218.74 | 219.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 218.99 | 218.74 | 219.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 218.99 | 218.74 | 219.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:45:00 | 219.13 | 218.74 | 219.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 218.45 | 218.68 | 219.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:30:00 | 217.82 | 218.46 | 218.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 10:00:00 | 218.01 | 218.33 | 218.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 13:15:00 | 221.01 | 219.02 | 218.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 221.01 | 219.02 | 218.96 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 217.36 | 218.79 | 218.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 213.33 | 217.41 | 218.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 212.85 | 212.55 | 214.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:45:00 | 213.58 | 212.55 | 214.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 215.82 | 212.92 | 213.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 215.82 | 212.92 | 213.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 216.29 | 213.60 | 214.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 216.29 | 213.60 | 214.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 217.75 | 214.43 | 214.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 218.20 | 215.18 | 214.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 239.95 | 240.78 | 237.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 14:45:00 | 239.44 | 240.78 | 237.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 240.69 | 240.66 | 238.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 243.30 | 241.25 | 238.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 235.34 | 238.41 | 238.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 235.34 | 238.41 | 238.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 232.30 | 236.96 | 237.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 225.30 | 225.03 | 227.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 09:15:00 | 224.29 | 225.03 | 227.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 228.29 | 224.65 | 225.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:00:00 | 228.29 | 224.65 | 225.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 226.82 | 225.08 | 226.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 226.35 | 225.16 | 225.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 226.58 | 225.75 | 225.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 226.71 | 226.07 | 226.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 226.71 | 226.07 | 226.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 229.48 | 227.01 | 226.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 12:15:00 | 228.10 | 228.20 | 227.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 13:00:00 | 228.10 | 228.20 | 227.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 230.95 | 229.60 | 228.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:15:00 | 233.01 | 229.60 | 228.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 233.04 | 231.27 | 229.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 226.15 | 230.30 | 230.26 | SL hit (close<static) qty=1.00 sl=228.32 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 224.60 | 229.16 | 229.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 224.04 | 227.37 | 228.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 222.18 | 221.76 | 223.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 10:30:00 | 222.59 | 221.76 | 223.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 208.03 | 207.29 | 208.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:30:00 | 207.06 | 207.29 | 208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 207.00 | 207.23 | 208.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 208.19 | 207.23 | 208.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 206.08 | 204.89 | 205.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 206.08 | 204.89 | 205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 205.98 | 205.11 | 205.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 13:30:00 | 204.50 | 204.94 | 205.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 194.27 | 196.74 | 199.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 195.18 | 195.02 | 197.40 | SL hit (close>ema200) qty=0.50 sl=195.02 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 208.30 | 200.33 | 199.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 214.50 | 207.62 | 203.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 210.20 | 210.81 | 207.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 210.20 | 210.81 | 207.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 206.90 | 210.03 | 207.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 206.90 | 210.03 | 207.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 206.00 | 209.23 | 207.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 206.00 | 209.23 | 207.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 204.52 | 207.37 | 206.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 204.52 | 207.37 | 206.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 204.20 | 206.74 | 206.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 204.20 | 206.74 | 206.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 204.20 | 206.23 | 206.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 203.39 | 205.66 | 206.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 205.73 | 205.67 | 206.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 205.73 | 205.67 | 206.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 204.73 | 205.49 | 205.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 205.98 | 205.49 | 205.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 198.26 | 203.08 | 204.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:00:00 | 197.95 | 202.05 | 203.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 197.97 | 201.12 | 203.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 188.05 | 191.46 | 195.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 188.07 | 191.46 | 195.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 187.94 | 186.85 | 189.15 | SL hit (close>ema200) qty=0.50 sl=186.85 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 190.37 | 188.23 | 188.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 191.44 | 189.19 | 188.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 197.51 | 197.69 | 194.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 197.09 | 197.69 | 194.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 195.04 | 196.56 | 195.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 195.04 | 196.56 | 195.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 194.15 | 196.08 | 195.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 194.15 | 196.08 | 195.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 193.95 | 195.65 | 194.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 193.00 | 195.65 | 194.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 193.61 | 194.95 | 194.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 193.61 | 194.95 | 194.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 194.39 | 194.79 | 194.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:00:00 | 194.39 | 194.79 | 194.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 195.00 | 194.83 | 194.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 196.90 | 195.33 | 194.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 207.03 | 208.58 | 208.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 207.03 | 208.58 | 208.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 206.71 | 207.81 | 208.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 206.48 | 206.34 | 207.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 11:00:00 | 206.48 | 206.34 | 207.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 204.73 | 206.02 | 206.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 203.40 | 205.80 | 206.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 203.17 | 204.31 | 205.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 193.23 | 197.21 | 199.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 193.01 | 195.81 | 198.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 196.19 | 195.03 | 197.40 | SL hit (close>ema200) qty=0.50 sl=195.03 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 199.84 | 195.61 | 195.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 200.73 | 196.63 | 196.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 11:15:00 | 196.84 | 197.82 | 197.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 11:15:00 | 196.84 | 197.82 | 197.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 196.84 | 197.82 | 197.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 196.84 | 197.82 | 197.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 197.53 | 197.76 | 197.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 15:00:00 | 198.20 | 197.78 | 197.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 195.86 | 197.37 | 197.19 | SL hit (close<static) qty=1.00 sl=196.60 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 195.79 | 197.06 | 197.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 194.50 | 196.55 | 196.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 191.13 | 190.78 | 192.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:00:00 | 191.13 | 190.78 | 192.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 191.45 | 190.86 | 191.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 190.21 | 190.86 | 191.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 190.21 | 190.73 | 191.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 189.88 | 190.41 | 191.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 15:15:00 | 191.20 | 190.95 | 190.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 15:15:00 | 191.20 | 190.95 | 190.94 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 188.86 | 190.53 | 190.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 187.01 | 189.83 | 190.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 188.04 | 186.95 | 188.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 188.04 | 186.95 | 188.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 188.04 | 186.95 | 188.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 189.00 | 186.95 | 188.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 190.45 | 187.36 | 187.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 190.45 | 187.36 | 187.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 188.53 | 187.59 | 187.88 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 190.98 | 188.63 | 188.32 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 185.70 | 188.34 | 188.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 185.37 | 187.75 | 188.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 177.94 | 175.93 | 178.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 177.94 | 175.93 | 178.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 178.76 | 176.47 | 177.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 178.76 | 176.47 | 177.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 179.46 | 177.07 | 177.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 179.46 | 177.07 | 177.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 176.45 | 177.39 | 177.86 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 180.46 | 178.18 | 178.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 182.37 | 180.20 | 179.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 180.10 | 180.18 | 179.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 180.10 | 180.18 | 179.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 180.67 | 180.33 | 179.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 179.64 | 180.33 | 179.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 180.39 | 180.56 | 179.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 180.11 | 180.56 | 179.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 181.43 | 180.74 | 180.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:45:00 | 182.49 | 181.37 | 180.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:30:00 | 182.73 | 181.68 | 180.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 182.58 | 181.74 | 180.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 182.65 | 181.51 | 180.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 181.83 | 181.58 | 181.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 181.18 | 181.58 | 181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 181.72 | 181.60 | 181.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 179.82 | 181.25 | 181.00 | SL hit (close<static) qty=1.00 sl=180.06 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 177.03 | 180.26 | 180.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 175.04 | 179.22 | 180.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 176.64 | 176.52 | 178.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 176.64 | 176.52 | 178.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 178.79 | 177.18 | 178.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 178.79 | 177.18 | 178.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 179.09 | 177.56 | 178.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 178.67 | 177.56 | 178.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 179.50 | 177.95 | 178.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 179.45 | 177.95 | 178.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 176.09 | 178.06 | 178.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 172.25 | 176.63 | 177.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 163.64 | 169.81 | 173.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 169.88 | 168.97 | 171.73 | SL hit (close>ema200) qty=0.50 sl=168.97 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 174.42 | 169.42 | 169.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 176.85 | 170.91 | 169.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 176.07 | 176.16 | 173.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 176.55 | 176.16 | 173.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 175.75 | 175.95 | 174.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 176.90 | 175.39 | 174.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 176.80 | 175.83 | 175.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 171.43 | 176.83 | 177.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 171.43 | 176.83 | 177.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 169.00 | 172.12 | 174.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 163.59 | 160.86 | 163.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 163.59 | 160.86 | 163.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 163.59 | 160.86 | 163.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 163.24 | 160.86 | 163.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 164.17 | 161.53 | 163.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 164.17 | 161.53 | 163.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 164.17 | 162.05 | 163.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 164.17 | 162.05 | 163.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 164.23 | 162.70 | 163.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 164.23 | 162.70 | 163.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 162.67 | 162.70 | 163.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:00:00 | 161.08 | 162.20 | 162.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 160.12 | 162.18 | 162.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 161.05 | 160.33 | 160.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 164.37 | 161.27 | 161.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 164.37 | 161.27 | 161.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 166.09 | 163.76 | 162.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 164.70 | 165.60 | 164.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 164.70 | 165.60 | 164.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 164.70 | 165.60 | 164.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 164.70 | 165.60 | 164.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 165.63 | 165.60 | 164.58 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 162.76 | 164.09 | 164.22 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 165.20 | 164.44 | 164.36 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 163.40 | 164.27 | 164.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 161.45 | 163.71 | 164.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 159.53 | 159.36 | 161.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 159.53 | 159.36 | 161.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 158.00 | 156.40 | 158.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 157.17 | 156.40 | 158.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 154.43 | 156.01 | 157.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 154.21 | 155.70 | 157.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 153.86 | 155.74 | 156.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 158.72 | 155.66 | 155.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 158.72 | 155.66 | 155.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 160.89 | 158.18 | 156.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 159.71 | 160.28 | 158.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:45:00 | 159.77 | 160.28 | 158.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 158.16 | 159.83 | 158.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 158.16 | 159.83 | 158.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 158.03 | 159.47 | 158.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 158.03 | 159.47 | 158.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 156.74 | 158.20 | 158.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 156.27 | 157.81 | 158.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 156.25 | 156.19 | 157.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 156.25 | 156.19 | 157.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 156.25 | 156.19 | 157.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 156.25 | 156.19 | 157.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 156.76 | 156.08 | 156.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 156.26 | 156.08 | 156.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 157.29 | 156.33 | 156.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 159.04 | 156.33 | 156.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 159.53 | 156.97 | 157.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 161.46 | 156.97 | 157.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 158.89 | 157.35 | 157.18 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 157.70 | 158.16 | 158.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 156.52 | 157.66 | 157.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 158.80 | 157.77 | 157.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 158.80 | 157.77 | 157.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 158.80 | 157.77 | 157.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 158.80 | 157.77 | 157.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 160.05 | 158.23 | 158.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 160.69 | 158.72 | 158.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 180.40 | 180.44 | 176.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 180.48 | 180.44 | 176.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 177.50 | 178.90 | 177.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 178.85 | 178.90 | 177.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 177.80 | 178.68 | 177.34 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 174.19 | 176.62 | 176.75 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 178.28 | 176.90 | 176.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 180.32 | 177.58 | 177.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 15:15:00 | 182.30 | 182.38 | 180.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:15:00 | 183.83 | 182.38 | 180.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 183.33 | 182.57 | 180.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:45:00 | 186.20 | 184.08 | 182.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 178.80 | 182.82 | 183.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 178.80 | 182.82 | 183.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 177.21 | 180.67 | 182.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 170.30 | 169.80 | 174.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 172.56 | 169.80 | 174.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 171.06 | 170.05 | 173.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 170.06 | 170.54 | 173.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 169.10 | 172.56 | 173.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 170.09 | 172.07 | 173.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:45:00 | 170.36 | 171.11 | 172.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 170.17 | 169.89 | 171.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 178.70 | 172.76 | 172.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 178.70 | 172.76 | 172.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 179.43 | 175.05 | 173.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 177.41 | 177.50 | 175.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:30:00 | 177.51 | 177.50 | 175.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 191.10 | 192.92 | 191.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 191.10 | 192.92 | 191.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 191.59 | 192.65 | 191.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 190.50 | 192.65 | 191.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 194.00 | 193.13 | 192.05 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 186.35 | 191.26 | 191.76 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 190.90 | 190.00 | 189.94 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 189.01 | 189.85 | 189.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 188.04 | 189.49 | 189.75 | Break + close below crossover candle low |

### Cycle 65 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 191.98 | 189.99 | 189.95 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 188.96 | 189.91 | 189.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 187.50 | 189.21 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 189.37 | 189.24 | 189.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 189.37 | 189.24 | 189.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 189.37 | 189.24 | 189.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 189.11 | 189.24 | 189.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 190.24 | 189.35 | 189.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 190.71 | 189.35 | 189.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 190.11 | 189.50 | 189.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 190.11 | 189.50 | 189.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 191.05 | 189.81 | 189.70 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 188.70 | 189.59 | 189.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 187.45 | 188.95 | 189.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 189.40 | 188.00 | 188.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 189.40 | 188.00 | 188.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 189.40 | 188.00 | 188.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 189.40 | 188.00 | 188.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 189.86 | 188.37 | 188.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 190.10 | 188.37 | 188.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 189.87 | 188.93 | 188.88 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 188.13 | 188.87 | 188.88 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 189.01 | 188.90 | 188.89 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 186.41 | 188.40 | 188.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 185.73 | 187.87 | 188.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 187.10 | 183.56 | 184.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 187.10 | 183.56 | 184.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 187.10 | 183.56 | 184.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 187.10 | 183.56 | 184.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 188.01 | 185.73 | 185.69 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 14:15:00 | 183.35 | 185.92 | 186.10 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 186.59 | 185.77 | 185.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 189.99 | 186.93 | 186.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 189.87 | 190.87 | 189.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 189.87 | 190.87 | 189.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 189.87 | 190.87 | 189.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 192.10 | 190.14 | 189.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 192.10 | 190.35 | 189.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:00:00 | 191.53 | 190.59 | 190.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:15:00 | 191.97 | 190.77 | 190.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 192.50 | 191.42 | 190.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 190.90 | 191.42 | 190.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 191.20 | 191.60 | 190.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 191.67 | 191.60 | 190.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 192.20 | 191.72 | 191.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 193.68 | 191.52 | 191.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:15:00 | 191.83 | 192.51 | 192.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 192.41 | 192.49 | 192.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 193.47 | 192.49 | 192.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 192.81 | 194.39 | 193.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:45:00 | 193.03 | 194.15 | 193.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 192.02 | 193.54 | 193.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 192.02 | 193.54 | 193.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 191.05 | 192.59 | 193.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 191.45 | 191.12 | 191.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 191.45 | 191.12 | 191.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 191.45 | 191.12 | 191.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 191.55 | 191.12 | 191.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 190.93 | 190.71 | 191.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 190.42 | 190.92 | 191.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 194.10 | 191.67 | 191.75 | SL hit (close>static) qty=1.00 sl=191.69 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 194.14 | 192.16 | 191.96 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 187.40 | 191.36 | 191.77 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 191.10 | 190.85 | 190.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 191.39 | 190.96 | 190.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 196.69 | 197.95 | 196.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 196.69 | 197.95 | 196.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 196.69 | 197.95 | 196.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 196.96 | 197.95 | 196.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 195.11 | 197.38 | 196.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 195.11 | 197.38 | 196.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 194.03 | 196.71 | 195.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 194.03 | 196.71 | 195.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 192.35 | 194.96 | 195.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 189.03 | 193.31 | 194.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 191.36 | 191.10 | 192.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 191.36 | 191.10 | 192.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 189.28 | 190.83 | 192.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 188.21 | 190.16 | 190.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 186.35 | 184.04 | 183.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 186.35 | 184.04 | 183.83 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 15:15:00 | 183.80 | 184.51 | 184.55 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 186.30 | 184.85 | 184.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 186.90 | 185.47 | 185.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 189.28 | 190.04 | 188.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 189.28 | 190.04 | 188.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 189.28 | 190.04 | 188.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 189.55 | 190.04 | 188.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 189.60 | 190.08 | 189.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 189.60 | 190.08 | 189.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 189.90 | 190.04 | 189.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 190.61 | 190.04 | 189.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 188.84 | 189.80 | 189.41 | SL hit (close<static) qty=1.00 sl=189.43 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 188.12 | 192.22 | 192.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 186.42 | 190.36 | 191.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 185.82 | 185.78 | 187.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 15:00:00 | 185.82 | 185.78 | 187.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 183.53 | 183.40 | 184.18 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 184.88 | 184.27 | 184.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 12:15:00 | 185.57 | 184.71 | 184.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 184.12 | 184.78 | 184.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 184.12 | 184.78 | 184.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 184.12 | 184.78 | 184.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 184.12 | 184.78 | 184.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 184.18 | 184.66 | 184.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 183.45 | 184.66 | 184.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 185.20 | 184.75 | 184.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 185.86 | 184.77 | 184.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 184.10 | 184.78 | 184.69 | SL hit (close<static) qty=1.00 sl=184.21 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 184.00 | 184.62 | 184.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 183.54 | 183.95 | 184.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 184.11 | 183.98 | 184.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 13:15:00 | 184.11 | 183.98 | 184.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 184.11 | 183.98 | 184.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 184.10 | 183.98 | 184.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 184.09 | 184.00 | 184.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:30:00 | 184.40 | 184.00 | 184.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 185.19 | 184.22 | 184.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 185.36 | 184.22 | 184.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 186.14 | 184.61 | 184.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 186.37 | 185.53 | 184.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 185.30 | 187.04 | 186.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 185.30 | 187.04 | 186.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 185.30 | 187.04 | 186.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 185.30 | 187.04 | 186.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 186.11 | 186.85 | 186.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:45:00 | 186.60 | 186.88 | 186.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:00:00 | 186.68 | 186.84 | 186.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 184.56 | 186.39 | 186.27 | SL hit (close<static) qty=1.00 sl=185.02 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 183.65 | 185.84 | 186.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 181.49 | 184.24 | 185.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 181.76 | 181.47 | 182.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 181.76 | 181.47 | 182.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 182.90 | 181.75 | 182.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 182.90 | 181.75 | 182.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 182.37 | 181.88 | 182.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 182.65 | 181.88 | 182.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 183.05 | 182.11 | 182.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 182.84 | 182.11 | 182.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 183.20 | 182.33 | 182.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 180.90 | 182.33 | 182.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 10:15:00 | 171.85 | 173.91 | 175.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 169.45 | 168.95 | 170.46 | SL hit (close>ema200) qty=0.50 sl=168.95 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 171.31 | 170.86 | 170.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 172.13 | 171.10 | 170.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 172.64 | 173.62 | 172.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 172.64 | 173.62 | 172.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 172.64 | 173.62 | 172.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 172.64 | 173.62 | 172.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 173.33 | 173.56 | 172.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 12:45:00 | 173.47 | 173.48 | 172.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 173.80 | 173.46 | 173.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 173.65 | 173.56 | 173.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 173.67 | 173.58 | 173.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 173.52 | 173.81 | 173.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 173.35 | 173.81 | 173.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 173.46 | 173.74 | 173.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 172.97 | 173.74 | 173.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 173.22 | 173.63 | 173.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 173.20 | 173.63 | 173.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 172.50 | 173.41 | 173.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 172.50 | 173.41 | 173.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 173.59 | 173.59 | 173.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 173.59 | 173.59 | 173.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 174.24 | 173.75 | 173.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:45:00 | 174.98 | 174.04 | 173.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 174.98 | 174.25 | 173.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 174.83 | 174.24 | 173.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 176.05 | 176.56 | 176.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 176.05 | 176.56 | 176.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 174.90 | 175.85 | 176.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 172.37 | 171.90 | 172.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 172.37 | 171.90 | 172.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 173.50 | 172.22 | 173.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 173.90 | 172.22 | 173.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 173.30 | 172.43 | 173.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 173.30 | 172.43 | 173.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 173.33 | 172.61 | 173.06 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 174.80 | 173.38 | 173.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 176.04 | 174.62 | 173.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 178.23 | 178.92 | 177.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 178.23 | 178.92 | 177.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 177.90 | 178.72 | 177.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 177.79 | 178.72 | 177.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 178.00 | 178.57 | 177.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:30:00 | 177.35 | 178.57 | 177.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 176.54 | 178.11 | 177.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 176.54 | 178.11 | 177.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 175.06 | 177.50 | 177.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 175.06 | 177.50 | 177.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 174.50 | 176.90 | 177.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 173.53 | 175.16 | 176.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 172.90 | 172.89 | 173.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:30:00 | 173.16 | 172.89 | 173.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 173.85 | 173.20 | 173.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 173.85 | 173.20 | 173.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 173.30 | 173.22 | 173.53 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 174.76 | 173.78 | 173.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 178.70 | 174.77 | 174.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 178.35 | 178.67 | 177.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 178.35 | 178.67 | 177.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 180.56 | 181.20 | 180.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 180.56 | 181.20 | 180.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 180.84 | 181.12 | 180.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 181.62 | 181.06 | 180.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 181.69 | 181.19 | 180.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 181.75 | 181.25 | 180.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 181.61 | 181.31 | 181.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 181.57 | 181.44 | 181.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 180.44 | 181.05 | 181.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 180.44 | 181.05 | 181.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 178.71 | 180.58 | 180.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 174.81 | 173.18 | 174.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 174.81 | 173.18 | 174.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 174.81 | 173.18 | 174.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 174.81 | 173.18 | 174.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 175.38 | 173.62 | 174.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 174.75 | 173.62 | 174.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 174.64 | 174.19 | 174.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 176.09 | 174.57 | 174.81 | SL hit (close>static) qty=1.00 sl=175.80 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 176.11 | 175.19 | 175.07 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 174.46 | 175.31 | 175.38 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 176.26 | 175.44 | 175.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 177.40 | 175.83 | 175.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 176.06 | 176.28 | 175.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 176.06 | 176.28 | 175.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 176.06 | 176.28 | 175.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 175.77 | 176.28 | 175.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 176.08 | 176.24 | 175.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 176.48 | 176.30 | 175.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 174.40 | 178.52 | 178.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 174.40 | 178.52 | 178.84 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 179.92 | 178.22 | 178.05 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 176.82 | 178.01 | 178.07 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 179.13 | 177.99 | 177.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 180.09 | 178.91 | 178.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 180.02 | 180.03 | 179.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 180.02 | 180.03 | 179.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 179.92 | 180.59 | 180.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 179.92 | 180.59 | 180.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 179.80 | 180.43 | 180.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 180.25 | 180.36 | 180.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 179.44 | 180.09 | 180.08 | SL hit (close<static) qty=1.00 sl=179.50 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 179.50 | 179.97 | 180.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 179.05 | 179.79 | 179.94 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 183.55 | 180.01 | 179.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 184.85 | 180.98 | 180.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 183.69 | 183.72 | 182.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:45:00 | 183.13 | 183.72 | 182.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 182.90 | 183.49 | 182.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 183.06 | 183.49 | 182.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 182.20 | 183.17 | 182.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:15:00 | 181.50 | 183.17 | 182.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 181.94 | 182.92 | 182.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 182.34 | 182.92 | 182.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 184.55 | 183.59 | 183.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 185.22 | 183.59 | 183.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 182.42 | 183.36 | 182.98 | SL hit (close<static) qty=1.00 sl=182.72 alert=retest2 |

### Cycle 104 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 181.61 | 182.96 | 183.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 180.99 | 182.24 | 182.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 179.65 | 179.50 | 180.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 179.65 | 179.50 | 180.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 180.69 | 179.83 | 180.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 180.69 | 179.83 | 180.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 180.50 | 179.96 | 180.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 180.58 | 179.96 | 180.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 180.37 | 180.05 | 180.53 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 181.20 | 180.83 | 180.78 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 180.50 | 180.78 | 180.79 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 181.94 | 181.01 | 180.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 182.90 | 181.97 | 181.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 10:15:00 | 181.86 | 181.95 | 181.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 11:00:00 | 181.86 | 181.95 | 181.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 182.77 | 183.41 | 182.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 184.92 | 183.26 | 182.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:45:00 | 183.97 | 183.81 | 183.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 184.21 | 184.17 | 183.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 184.30 | 184.21 | 183.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 184.29 | 184.26 | 184.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 183.26 | 184.03 | 184.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 183.26 | 184.03 | 184.04 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 184.10 | 184.04 | 184.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 184.35 | 184.10 | 184.07 | Break + close above crossover candle high |

### Cycle 110 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 183.00 | 184.12 | 184.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 182.69 | 183.84 | 184.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 184.18 | 183.76 | 183.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 184.18 | 183.76 | 183.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 184.18 | 183.76 | 183.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 184.18 | 183.76 | 183.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 182.92 | 183.59 | 183.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 182.25 | 183.13 | 183.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 184.68 | 182.51 | 182.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 184.68 | 182.51 | 182.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 185.05 | 183.01 | 182.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 183.95 | 184.04 | 183.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:45:00 | 184.21 | 184.04 | 183.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 184.00 | 184.03 | 183.41 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 174.03 | 181.91 | 182.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 172.00 | 174.64 | 175.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 170.95 | 170.81 | 172.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 09:15:00 | 170.70 | 170.81 | 172.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 168.20 | 167.67 | 168.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 168.20 | 167.67 | 168.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 167.90 | 167.72 | 168.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 168.60 | 167.72 | 168.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 169.04 | 167.98 | 168.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 169.02 | 167.98 | 168.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 169.17 | 168.22 | 168.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 168.36 | 168.56 | 168.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 169.19 | 168.70 | 168.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 169.19 | 168.70 | 168.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 170.20 | 169.00 | 168.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 14:15:00 | 169.01 | 169.14 | 168.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 14:15:00 | 169.01 | 169.14 | 168.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 169.01 | 169.14 | 168.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 168.98 | 169.14 | 168.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 168.66 | 169.04 | 168.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 171.45 | 169.04 | 168.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:00:00 | 169.30 | 170.14 | 169.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 168.80 | 169.53 | 169.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 168.80 | 169.53 | 169.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 168.17 | 169.26 | 169.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 169.51 | 168.81 | 169.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 169.51 | 168.81 | 169.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 169.51 | 168.81 | 169.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 168.19 | 168.89 | 169.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 167.33 | 168.61 | 168.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 168.04 | 168.43 | 168.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 168.20 | 168.00 | 168.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 168.83 | 168.16 | 168.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 168.84 | 168.16 | 168.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 168.41 | 168.21 | 168.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 168.18 | 168.21 | 168.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 169.95 | 168.67 | 168.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 169.95 | 168.67 | 168.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 170.91 | 169.30 | 168.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 172.01 | 172.11 | 171.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:30:00 | 172.03 | 172.11 | 171.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 171.89 | 172.06 | 171.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 171.89 | 172.06 | 171.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 170.80 | 171.81 | 171.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 170.80 | 171.81 | 171.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 171.08 | 171.66 | 171.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 171.48 | 171.66 | 171.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 171.05 | 171.43 | 171.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 171.73 | 171.43 | 171.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 171.57 | 171.42 | 171.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 170.97 | 171.27 | 171.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 170.97 | 171.27 | 171.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 170.20 | 170.82 | 171.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 170.67 | 170.56 | 170.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 170.67 | 170.56 | 170.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 170.67 | 170.56 | 170.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 170.67 | 170.56 | 170.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 170.32 | 170.51 | 170.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 169.93 | 170.38 | 170.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 171.75 | 170.74 | 170.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 171.75 | 170.74 | 170.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 172.17 | 171.22 | 170.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 171.40 | 171.62 | 171.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 171.40 | 171.62 | 171.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 171.40 | 171.62 | 171.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 171.51 | 171.62 | 171.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 171.10 | 171.51 | 171.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 171.10 | 171.51 | 171.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 171.30 | 171.47 | 171.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 171.57 | 171.47 | 171.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 171.77 | 173.30 | 173.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 170.97 | 172.83 | 173.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 170.97 | 172.83 | 173.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 169.99 | 171.75 | 172.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 164.51 | 164.39 | 165.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:45:00 | 164.62 | 164.39 | 165.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 166.09 | 164.51 | 165.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 166.09 | 164.51 | 165.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 167.06 | 165.02 | 165.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 166.80 | 165.02 | 165.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 166.08 | 165.35 | 165.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 165.39 | 165.21 | 165.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 165.18 | 165.15 | 165.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 165.62 | 165.40 | 165.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 165.62 | 165.40 | 165.38 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 165.18 | 165.35 | 165.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 164.84 | 165.24 | 165.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 164.78 | 164.69 | 164.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 164.78 | 164.69 | 164.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 164.78 | 164.69 | 164.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 163.61 | 164.36 | 164.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:30:00 | 163.05 | 162.77 | 163.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 162.96 | 162.81 | 163.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 163.49 | 163.20 | 163.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 163.49 | 163.20 | 163.19 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 162.45 | 163.15 | 163.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 161.61 | 162.67 | 162.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 161.40 | 160.18 | 161.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 161.40 | 160.18 | 161.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 161.40 | 160.18 | 161.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 164.15 | 160.18 | 161.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 164.80 | 161.10 | 161.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 164.66 | 161.10 | 161.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 166.25 | 162.13 | 161.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 166.77 | 164.14 | 162.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 166.75 | 167.00 | 165.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 166.75 | 167.00 | 165.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 166.00 | 166.78 | 165.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 166.00 | 166.78 | 165.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 165.89 | 166.60 | 165.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 166.94 | 166.42 | 165.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 167.51 | 166.55 | 165.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 164.52 | 166.11 | 165.87 | SL hit (close<static) qty=1.00 sl=165.55 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 163.45 | 165.57 | 165.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 163.15 | 165.09 | 165.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 162.50 | 161.09 | 162.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 162.50 | 161.09 | 162.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 162.50 | 161.09 | 162.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 162.07 | 161.21 | 162.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 164.90 | 163.02 | 162.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 164.90 | 163.02 | 162.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 165.38 | 164.35 | 163.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 160.35 | 163.77 | 163.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 160.35 | 163.77 | 163.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 160.35 | 163.77 | 163.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 160.35 | 163.77 | 163.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 160.27 | 163.07 | 163.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 159.81 | 161.93 | 162.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 161.01 | 160.66 | 161.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 161.01 | 160.66 | 161.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 161.56 | 160.84 | 161.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 161.56 | 160.84 | 161.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 162.27 | 161.13 | 161.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 162.27 | 161.13 | 161.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 162.60 | 161.42 | 161.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 162.60 | 161.42 | 161.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 162.69 | 161.94 | 161.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 165.27 | 163.44 | 162.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 161.96 | 163.80 | 163.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 161.96 | 163.80 | 163.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 161.96 | 163.80 | 163.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 161.96 | 163.80 | 163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 162.45 | 163.53 | 163.33 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 162.53 | 163.15 | 163.18 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 163.50 | 163.25 | 163.22 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 162.76 | 163.15 | 163.18 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 11:15:00 | 163.93 | 163.34 | 163.26 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 161.55 | 163.07 | 163.19 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 164.24 | 162.98 | 162.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 164.70 | 163.32 | 163.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 167.05 | 167.33 | 166.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 167.05 | 167.33 | 166.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 166.80 | 167.13 | 166.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 167.59 | 167.13 | 166.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 167.99 | 167.30 | 166.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 169.64 | 167.64 | 166.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 169.44 | 167.53 | 167.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 168.91 | 168.52 | 167.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 168.99 | 169.14 | 168.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 169.08 | 169.13 | 168.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 169.42 | 169.30 | 168.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 168.40 | 169.24 | 168.85 | SL hit (close<static) qty=1.00 sl=168.44 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 166.06 | 168.87 | 168.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 164.44 | 167.08 | 168.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 156.72 | 156.48 | 159.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 156.72 | 156.48 | 159.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 155.57 | 156.23 | 157.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 157.13 | 156.23 | 157.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 152.39 | 150.77 | 151.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 154.50 | 150.77 | 151.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 151.08 | 150.83 | 151.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:45:00 | 149.98 | 150.73 | 151.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 153.00 | 150.09 | 150.77 | SL hit (close>static) qty=1.00 sl=152.49 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 153.28 | 151.25 | 151.22 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 148.58 | 150.90 | 151.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 148.22 | 150.18 | 150.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 147.40 | 146.59 | 147.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 147.40 | 146.59 | 147.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 146.78 | 146.63 | 147.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:00:00 | 146.20 | 146.54 | 147.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 149.04 | 147.62 | 147.75 | SL hit (close>static) qty=1.00 sl=148.22 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 149.78 | 148.06 | 147.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 150.39 | 148.52 | 148.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 146.71 | 149.18 | 148.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 146.71 | 149.18 | 148.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 146.71 | 149.18 | 148.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 146.71 | 149.18 | 148.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 147.34 | 148.81 | 148.58 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 146.80 | 148.41 | 148.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 145.41 | 147.81 | 148.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 146.29 | 146.26 | 147.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 146.29 | 146.26 | 147.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 146.29 | 146.26 | 147.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 146.49 | 146.26 | 147.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 140.60 | 137.89 | 139.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 140.60 | 137.89 | 139.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 138.66 | 138.59 | 139.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:45:00 | 137.75 | 138.57 | 139.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 13:45:00 | 138.21 | 138.22 | 138.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 138.09 | 138.22 | 138.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 138.09 | 137.90 | 138.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 138.97 | 138.11 | 138.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 138.76 | 138.11 | 138.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 140.01 | 138.49 | 138.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-30 11:15:00 | 140.01 | 138.49 | 138.58 | SL hit (close>static) qty=1.00 sl=139.39 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 139.37 | 138.67 | 138.66 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 137.67 | 138.54 | 138.60 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 140.21 | 138.76 | 138.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 142.85 | 141.40 | 140.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 152.17 | 152.66 | 150.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 152.17 | 152.66 | 150.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 153.85 | 153.25 | 151.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 155.90 | 153.51 | 152.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 162.26 | 165.28 | 165.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 162.26 | 165.28 | 165.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 161.37 | 164.50 | 165.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 166.83 | 164.06 | 164.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 166.83 | 164.06 | 164.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 166.83 | 164.06 | 164.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 166.83 | 164.06 | 164.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 166.02 | 164.94 | 164.82 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 163.60 | 164.66 | 164.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 163.01 | 164.33 | 164.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 164.67 | 164.18 | 164.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 164.67 | 164.18 | 164.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 164.67 | 164.18 | 164.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 163.84 | 164.18 | 164.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 165.51 | 164.56 | 164.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 165.51 | 164.56 | 164.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 165.89 | 164.83 | 164.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 166.67 | 166.84 | 166.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 11:45:00 | 166.67 | 166.84 | 166.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 166.29 | 166.73 | 166.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 166.29 | 166.73 | 166.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 166.14 | 166.61 | 166.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 166.09 | 166.61 | 166.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 166.51 | 166.59 | 166.18 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 199.10 | 2024-05-17 09:15:00 | 196.40 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-05-30 12:15:00 | 199.55 | 2024-05-31 12:15:00 | 203.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-30 13:00:00 | 199.75 | 2024-05-31 12:15:00 | 203.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-06-11 12:00:00 | 209.68 | 2024-06-19 13:15:00 | 218.15 | STOP_HIT | 1.00 | 4.04% |
| SELL | retest2 | 2024-06-20 13:15:00 | 217.50 | 2024-06-27 10:15:00 | 215.00 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2024-06-21 09:15:00 | 218.48 | 2024-06-27 10:15:00 | 215.00 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2024-07-08 14:15:00 | 222.27 | 2024-07-18 11:15:00 | 230.15 | STOP_HIT | 1.00 | 3.55% |
| SELL | retest2 | 2024-07-23 12:15:00 | 213.19 | 2024-07-25 10:15:00 | 228.89 | STOP_HIT | 1.00 | -7.36% |
| SELL | retest2 | 2024-07-24 11:15:00 | 221.09 | 2024-07-25 10:15:00 | 228.89 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-07-24 12:30:00 | 220.75 | 2024-07-25 10:15:00 | 228.89 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2024-07-29 13:15:00 | 231.82 | 2024-07-31 09:15:00 | 243.41 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-29 14:45:00 | 231.98 | 2024-07-31 09:15:00 | 243.58 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-29 15:15:00 | 232.00 | 2024-07-31 09:15:00 | 243.60 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-29 13:15:00 | 231.82 | 2024-08-01 11:15:00 | 238.30 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2024-07-29 14:45:00 | 231.98 | 2024-08-01 11:15:00 | 238.30 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2024-07-29 15:15:00 | 232.00 | 2024-08-01 11:15:00 | 238.30 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2024-08-09 10:15:00 | 228.33 | 2024-08-12 11:15:00 | 231.11 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-12 09:45:00 | 228.42 | 2024-08-12 11:15:00 | 231.11 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-28 11:30:00 | 236.50 | 2024-08-28 15:15:00 | 234.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-09-11 10:00:00 | 219.08 | 2024-09-13 10:15:00 | 220.75 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-11 11:00:00 | 218.82 | 2024-09-13 10:15:00 | 220.75 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-09-12 13:00:00 | 218.40 | 2024-09-13 10:15:00 | 220.75 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-16 14:30:00 | 217.82 | 2024-09-17 13:15:00 | 221.01 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-17 10:00:00 | 218.01 | 2024-09-17 13:15:00 | 221.01 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-03 10:45:00 | 243.30 | 2024-10-04 10:15:00 | 235.34 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-10-10 11:30:00 | 226.35 | 2024-10-11 11:15:00 | 226.71 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-10-11 10:15:00 | 226.58 | 2024-10-11 11:15:00 | 226.71 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-10-15 10:15:00 | 233.01 | 2024-10-17 09:15:00 | 226.15 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-10-16 09:15:00 | 233.04 | 2024-10-17 09:15:00 | 226.15 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-10-30 13:30:00 | 204.50 | 2024-11-05 09:15:00 | 194.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-30 13:30:00 | 204.50 | 2024-11-05 13:15:00 | 195.18 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2024-11-12 11:00:00 | 197.95 | 2024-11-14 09:15:00 | 188.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 197.97 | 2024-11-14 09:15:00 | 188.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:00:00 | 197.95 | 2024-11-19 09:15:00 | 187.94 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2024-11-12 11:45:00 | 197.97 | 2024-11-19 09:15:00 | 187.94 | STOP_HIT | 0.50 | 5.07% |
| BUY | retest2 | 2024-11-28 09:30:00 | 196.90 | 2024-12-10 12:15:00 | 207.03 | STOP_HIT | 1.00 | 5.14% |
| SELL | retest2 | 2024-12-13 09:15:00 | 203.40 | 2024-12-18 12:15:00 | 193.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 203.17 | 2024-12-18 14:15:00 | 193.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 203.40 | 2024-12-19 11:15:00 | 196.19 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2024-12-16 09:15:00 | 203.17 | 2024-12-19 11:15:00 | 196.19 | STOP_HIT | 0.50 | 3.44% |
| BUY | retest2 | 2024-12-26 15:00:00 | 198.20 | 2024-12-27 09:15:00 | 195.86 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-02 10:30:00 | 189.88 | 2025-01-03 15:15:00 | 191.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-20 12:45:00 | 182.49 | 2025-01-21 14:15:00 | 179.82 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-01-20 14:30:00 | 182.73 | 2025-01-21 14:15:00 | 179.82 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-21 09:15:00 | 182.58 | 2025-01-21 14:15:00 | 179.82 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-01-21 11:45:00 | 182.65 | 2025-01-21 14:15:00 | 179.82 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-01-27 09:15:00 | 172.25 | 2025-01-28 09:15:00 | 163.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 172.25 | 2025-01-28 12:15:00 | 169.88 | STOP_HIT | 0.50 | 1.38% |
| BUY | retest2 | 2025-02-04 09:30:00 | 176.90 | 2025-02-07 09:15:00 | 171.43 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-02-04 10:45:00 | 176.80 | 2025-02-07 09:15:00 | 171.43 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-02-14 14:00:00 | 161.08 | 2025-02-18 14:15:00 | 164.37 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-02-17 09:15:00 | 160.12 | 2025-02-18 14:15:00 | 164.37 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-02-18 11:00:00 | 161.05 | 2025-02-18 14:15:00 | 164.37 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-03-03 11:15:00 | 154.21 | 2025-03-05 11:15:00 | 158.72 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-03-03 15:15:00 | 153.86 | 2025-03-05 11:15:00 | 158.72 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-04-01 13:45:00 | 186.20 | 2025-04-04 09:15:00 | 178.80 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-04-08 10:30:00 | 170.06 | 2025-04-15 09:15:00 | 178.70 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-04-09 09:15:00 | 169.10 | 2025-04-15 09:15:00 | 178.70 | STOP_HIT | 1.00 | -5.68% |
| SELL | retest2 | 2025-04-09 10:00:00 | 170.09 | 2025-04-15 09:15:00 | 178.70 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2025-04-09 11:45:00 | 170.36 | 2025-04-15 09:15:00 | 178.70 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-05-22 09:15:00 | 192.10 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-22 10:15:00 | 192.10 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-22 11:00:00 | 191.53 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-05-22 12:15:00 | 191.97 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-23 11:15:00 | 191.67 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-05-23 12:00:00 | 192.20 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-05-26 09:15:00 | 193.68 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-27 10:15:00 | 191.83 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-05-27 11:15:00 | 193.47 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-05-29 10:00:00 | 192.81 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-05-29 10:45:00 | 193.03 | 2025-05-29 13:15:00 | 192.02 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-06-02 14:30:00 | 190.42 | 2025-06-03 09:15:00 | 194.10 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-06-18 10:15:00 | 188.21 | 2025-06-24 09:15:00 | 186.35 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-02 09:15:00 | 190.61 | 2025-07-02 09:15:00 | 188.84 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-02 11:00:00 | 190.80 | 2025-07-09 09:15:00 | 188.12 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-02 14:00:00 | 190.80 | 2025-07-09 09:15:00 | 188.12 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-18 14:15:00 | 185.86 | 2025-07-21 09:15:00 | 184.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-25 11:45:00 | 186.60 | 2025-07-25 13:15:00 | 184.56 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-25 13:00:00 | 186.68 | 2025-07-25 13:15:00 | 184.56 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-07-30 09:15:00 | 180.90 | 2025-08-05 10:15:00 | 171.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:15:00 | 180.90 | 2025-08-07 14:15:00 | 169.45 | STOP_HIT | 0.50 | 6.33% |
| BUY | retest2 | 2025-08-13 12:45:00 | 173.47 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-08-13 15:15:00 | 173.80 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-08-14 10:00:00 | 173.65 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-08-14 11:00:00 | 173.67 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2025-08-19 10:45:00 | 174.98 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-08-19 13:15:00 | 174.98 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-08-19 14:15:00 | 174.83 | 2025-08-25 10:15:00 | 176.05 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-09-19 09:15:00 | 181.62 | 2025-09-23 09:15:00 | 180.44 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-19 10:00:00 | 181.69 | 2025-09-23 09:15:00 | 180.44 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-19 11:15:00 | 181.75 | 2025-09-23 09:15:00 | 180.44 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-19 15:00:00 | 181.61 | 2025-09-23 09:15:00 | 180.44 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-29 11:15:00 | 174.75 | 2025-09-29 13:15:00 | 176.09 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-29 12:30:00 | 174.64 | 2025-09-29 13:15:00 | 176.09 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-06 14:30:00 | 176.48 | 2025-10-14 13:15:00 | 174.40 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-27 15:15:00 | 180.25 | 2025-10-28 10:15:00 | 179.44 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-10-31 14:15:00 | 185.22 | 2025-10-31 14:15:00 | 182.42 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-17 09:15:00 | 184.92 | 2025-11-20 09:15:00 | 183.26 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-17 11:45:00 | 183.97 | 2025-11-20 09:15:00 | 183.26 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-11-18 12:00:00 | 184.21 | 2025-11-20 09:15:00 | 183.26 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-18 15:00:00 | 184.30 | 2025-11-20 09:15:00 | 183.26 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-24 10:45:00 | 182.25 | 2025-11-26 11:15:00 | 184.68 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-10 14:00:00 | 168.36 | 2025-12-11 10:15:00 | 169.19 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-12 09:15:00 | 171.45 | 2025-12-16 10:15:00 | 168.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-15 10:00:00 | 169.30 | 2025-12-16 10:15:00 | 168.80 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-17 14:30:00 | 168.19 | 2025-12-19 14:15:00 | 169.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-18 09:30:00 | 167.33 | 2025-12-19 14:15:00 | 169.95 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-18 12:15:00 | 168.04 | 2025-12-19 14:15:00 | 169.95 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-19 10:00:00 | 168.20 | 2025-12-19 14:15:00 | 169.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-19 13:00:00 | 168.18 | 2025-12-19 14:15:00 | 169.95 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-26 11:15:00 | 171.73 | 2025-12-26 14:15:00 | 170.97 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-12-26 12:45:00 | 171.57 | 2025-12-26 14:15:00 | 170.97 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-12-30 11:45:00 | 169.93 | 2025-12-31 09:15:00 | 171.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-01 14:15:00 | 171.57 | 2026-01-06 10:15:00 | 170.97 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-01-06 09:30:00 | 171.77 | 2026-01-06 10:15:00 | 170.97 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-01-13 12:45:00 | 165.39 | 2026-01-14 12:15:00 | 165.62 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-01-14 10:00:00 | 165.18 | 2026-01-14 12:15:00 | 165.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-01-20 09:15:00 | 163.61 | 2026-01-22 14:15:00 | 163.49 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-01-21 13:30:00 | 163.05 | 2026-01-22 14:15:00 | 163.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-01-22 09:45:00 | 162.96 | 2026-01-22 14:15:00 | 163.49 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-01-30 14:15:00 | 166.94 | 2026-02-01 10:15:00 | 164.52 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-30 14:45:00 | 167.51 | 2026-02-01 10:15:00 | 164.52 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-02-03 10:30:00 | 162.07 | 2026-02-04 09:15:00 | 164.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-20 10:30:00 | 169.64 | 2026-02-27 09:15:00 | 168.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-02-25 09:15:00 | 169.44 | 2026-03-02 09:15:00 | 166.06 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-25 14:15:00 | 168.91 | 2026-03-02 09:15:00 | 166.06 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-26 12:30:00 | 168.99 | 2026-03-02 09:15:00 | 166.06 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-02-26 14:30:00 | 169.42 | 2026-03-02 09:15:00 | 166.06 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-27 10:30:00 | 169.70 | 2026-03-02 09:15:00 | 166.06 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-03-11 11:45:00 | 149.98 | 2026-03-12 10:15:00 | 153.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-03-17 12:00:00 | 146.20 | 2026-03-18 10:15:00 | 149.04 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-27 09:45:00 | 137.75 | 2026-03-30 11:15:00 | 140.01 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-27 13:45:00 | 138.21 | 2026-03-30 11:15:00 | 140.01 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-27 14:15:00 | 138.09 | 2026-03-30 11:15:00 | 140.01 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-30 10:15:00 | 138.09 | 2026-03-30 11:15:00 | 140.01 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-15 09:15:00 | 155.90 | 2026-04-30 09:15:00 | 162.26 | STOP_HIT | 1.00 | 4.08% |
| SELL | retest2 | 2026-05-06 12:45:00 | 163.84 | 2026-05-06 14:15:00 | 165.51 | STOP_HIT | 1.00 | -1.02% |
