# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 408.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 236 |
| ALERT1 | 152 |
| ALERT2 | 150 |
| ALERT2_SKIP | 76 |
| ALERT3 | 441 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 191 |
| PARTIAL | 26 |
| TARGET_HIT | 6 |
| STOP_HIT | 199 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 227 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 85 / 142
- **Target hits / Stop hits / Partials:** 6 / 195 / 26
- **Avg / median % per leg:** 0.62% / -0.32%
- **Sum % (uncompounded):** 141.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 99 | 22 | 22.2% | 4 | 93 | 2 | 0.02% | 2.0% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 0 | 8 | 2 | 0.86% | 8.6% |
| BUY @ 3rd Alert (retest2) | 89 | 18 | 20.2% | 4 | 85 | 0 | -0.07% | -6.6% |
| SELL (all) | 128 | 63 | 49.2% | 2 | 102 | 24 | 1.09% | 139.2% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.53% | -1.1% |
| SELL @ 3rd Alert (retest2) | 126 | 62 | 49.2% | 2 | 100 | 24 | 1.11% | 140.2% |
| retest1 (combined) | 12 | 5 | 41.7% | 0 | 10 | 2 | 0.63% | 7.5% |
| retest2 (combined) | 215 | 80 | 37.2% | 6 | 185 | 24 | 0.62% | 133.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 192.70 | 194.58 | 194.74 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 13:15:00 | 195.15 | 194.44 | 194.43 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 193.70 | 194.29 | 194.36 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 196.45 | 194.57 | 194.47 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 192.90 | 194.43 | 194.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 12:15:00 | 191.55 | 193.66 | 194.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 11:15:00 | 192.00 | 190.82 | 192.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 11:15:00 | 192.00 | 190.82 | 192.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 192.00 | 190.82 | 192.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:00:00 | 192.00 | 190.82 | 192.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 190.45 | 190.75 | 191.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:15:00 | 192.80 | 190.75 | 191.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 191.85 | 190.97 | 191.97 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 194.80 | 192.76 | 192.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 196.00 | 194.47 | 193.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 195.60 | 195.76 | 194.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 15:00:00 | 195.60 | 195.76 | 194.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 195.25 | 195.57 | 194.69 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 190.15 | 194.09 | 194.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 186.45 | 191.11 | 192.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 187.20 | 186.68 | 189.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 10:00:00 | 187.20 | 186.68 | 189.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 187.90 | 186.99 | 188.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 187.90 | 186.99 | 188.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 190.30 | 187.70 | 188.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:00:00 | 190.30 | 187.70 | 188.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 191.60 | 188.48 | 188.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 11:15:00 | 191.70 | 188.48 | 188.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 11:15:00 | 193.00 | 189.38 | 189.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 09:15:00 | 201.00 | 193.50 | 191.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 09:15:00 | 218.90 | 219.56 | 216.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 09:45:00 | 218.65 | 219.56 | 216.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 216.20 | 219.92 | 218.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 216.20 | 219.92 | 218.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 216.40 | 219.22 | 218.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 216.10 | 219.22 | 218.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 215.15 | 217.83 | 218.10 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 09:15:00 | 226.55 | 219.33 | 218.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 10:15:00 | 230.70 | 221.60 | 219.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 09:15:00 | 227.45 | 227.58 | 224.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-12 09:45:00 | 226.50 | 227.58 | 224.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 227.75 | 228.36 | 227.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 12:45:00 | 230.05 | 228.44 | 227.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 10:15:00 | 225.45 | 227.19 | 227.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 10:15:00 | 225.45 | 227.19 | 227.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 11:15:00 | 223.50 | 226.45 | 226.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 224.70 | 224.44 | 225.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 224.70 | 224.44 | 225.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 224.70 | 224.44 | 225.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:30:00 | 225.55 | 224.44 | 225.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 224.90 | 224.53 | 225.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:45:00 | 225.15 | 224.53 | 225.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 224.45 | 224.52 | 225.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 13:30:00 | 223.50 | 224.44 | 225.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 14:15:00 | 224.00 | 224.44 | 225.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 222.20 | 224.32 | 225.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 10:15:00 | 223.80 | 222.51 | 223.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 222.95 | 222.59 | 223.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 11:15:00 | 222.20 | 222.59 | 223.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 14:45:00 | 222.40 | 222.54 | 223.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 09:15:00 | 224.75 | 222.90 | 223.07 | SL hit (close>static) qty=1.00 sl=224.65 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 11:15:00 | 225.70 | 223.53 | 223.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 12:15:00 | 227.05 | 224.23 | 223.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 224.40 | 226.97 | 225.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 224.40 | 226.97 | 225.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 224.40 | 226.97 | 225.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:45:00 | 223.70 | 226.97 | 225.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 224.15 | 226.41 | 225.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 224.15 | 226.41 | 225.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 221.65 | 225.45 | 224.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 221.65 | 225.45 | 224.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 219.55 | 224.27 | 224.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 15:15:00 | 217.90 | 221.45 | 222.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 213.45 | 211.65 | 214.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 15:00:00 | 213.45 | 211.65 | 214.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 218.65 | 213.33 | 214.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 218.65 | 213.33 | 214.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 217.05 | 214.07 | 215.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 11:30:00 | 216.00 | 214.33 | 215.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 09:15:00 | 214.45 | 212.03 | 212.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 214.45 | 212.03 | 212.00 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 14:15:00 | 211.05 | 211.88 | 211.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 15:15:00 | 210.35 | 211.57 | 211.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 13:15:00 | 208.90 | 207.23 | 208.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 13:15:00 | 208.90 | 207.23 | 208.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 208.90 | 207.23 | 208.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:00:00 | 208.90 | 207.23 | 208.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 208.45 | 207.47 | 208.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:30:00 | 208.70 | 207.47 | 208.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 209.40 | 207.86 | 208.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 207.80 | 207.86 | 208.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 209.40 | 208.17 | 208.52 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 209.95 | 208.78 | 208.66 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 205.55 | 208.31 | 208.49 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 210.50 | 208.38 | 208.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 213.00 | 209.31 | 208.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 211.75 | 213.72 | 212.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 211.75 | 213.72 | 212.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 211.75 | 213.72 | 212.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:45:00 | 211.20 | 213.72 | 212.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 211.85 | 213.35 | 212.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:00:00 | 211.85 | 213.35 | 212.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 209.80 | 212.64 | 212.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:45:00 | 210.30 | 212.64 | 212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 210.00 | 211.84 | 211.84 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 15:15:00 | 212.15 | 211.55 | 211.53 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 09:15:00 | 211.15 | 211.47 | 211.49 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 213.90 | 211.89 | 211.68 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 208.00 | 210.96 | 211.33 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 212.65 | 210.52 | 210.25 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 210.75 | 211.29 | 211.30 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 14:15:00 | 211.50 | 211.33 | 211.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 15:15:00 | 213.50 | 211.77 | 211.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 218.20 | 218.84 | 216.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:30:00 | 218.00 | 218.84 | 216.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 218.40 | 218.68 | 217.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:30:00 | 217.05 | 218.68 | 217.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 217.25 | 218.82 | 217.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 217.25 | 218.82 | 217.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 216.20 | 218.29 | 217.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 216.90 | 218.29 | 217.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 216.15 | 217.86 | 217.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 216.15 | 217.86 | 217.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 212.45 | 216.78 | 217.05 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 218.50 | 216.76 | 216.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 220.40 | 217.85 | 217.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 10:15:00 | 219.80 | 221.10 | 219.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 10:15:00 | 219.80 | 221.10 | 219.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 219.80 | 221.10 | 219.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 219.80 | 221.10 | 219.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 220.30 | 220.94 | 219.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:30:00 | 219.75 | 220.94 | 219.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 215.65 | 219.88 | 219.15 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 14:15:00 | 215.95 | 218.60 | 218.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 15:15:00 | 215.00 | 217.88 | 218.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 14:15:00 | 215.45 | 214.93 | 216.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-08 15:00:00 | 215.45 | 214.93 | 216.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 220.10 | 215.98 | 216.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:15:00 | 221.65 | 215.98 | 216.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 220.75 | 216.93 | 216.94 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 11:15:00 | 222.35 | 218.01 | 217.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 13:15:00 | 227.40 | 220.46 | 218.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 10:15:00 | 246.75 | 248.21 | 241.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-14 10:30:00 | 245.75 | 248.21 | 241.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 244.85 | 248.96 | 245.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:00:00 | 244.85 | 248.96 | 245.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 245.35 | 248.24 | 245.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 12:30:00 | 247.50 | 248.09 | 245.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 10:15:00 | 243.85 | 246.51 | 245.82 | SL hit (close<static) qty=1.00 sl=244.60 alert=retest2 |

### Cycle 31 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 11:15:00 | 239.20 | 245.05 | 245.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 233.70 | 240.55 | 242.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 241.90 | 238.44 | 240.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 241.90 | 238.44 | 240.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 241.90 | 238.44 | 240.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 244.15 | 238.44 | 240.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 243.85 | 239.53 | 240.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 243.85 | 239.53 | 240.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 245.35 | 241.57 | 241.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 245.80 | 243.65 | 242.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 10:15:00 | 243.55 | 243.63 | 242.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 10:15:00 | 243.55 | 243.63 | 242.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 243.55 | 243.63 | 242.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:00:00 | 243.55 | 243.63 | 242.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 239.90 | 242.96 | 242.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:30:00 | 241.30 | 242.96 | 242.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 15:15:00 | 239.80 | 242.33 | 242.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 10:15:00 | 237.25 | 239.38 | 240.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 235.25 | 233.15 | 235.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 235.25 | 233.15 | 235.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 235.25 | 233.15 | 235.38 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 14:15:00 | 240.05 | 236.37 | 236.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 242.30 | 238.22 | 237.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 12:15:00 | 238.75 | 241.33 | 239.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 12:15:00 | 238.75 | 241.33 | 239.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 238.75 | 241.33 | 239.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:00:00 | 238.75 | 241.33 | 239.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 239.05 | 240.88 | 239.89 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 236.65 | 239.29 | 239.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 235.80 | 237.90 | 238.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 15:15:00 | 238.00 | 237.30 | 238.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 238.00 | 237.30 | 238.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 238.00 | 237.30 | 238.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 238.90 | 237.30 | 238.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 236.90 | 237.22 | 238.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:30:00 | 238.65 | 237.22 | 238.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 240.40 | 237.68 | 238.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:45:00 | 242.95 | 237.68 | 238.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 250.60 | 240.26 | 239.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 261.05 | 244.42 | 241.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 09:15:00 | 254.55 | 255.57 | 250.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 10:00:00 | 254.55 | 255.57 | 250.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 253.05 | 254.85 | 251.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:30:00 | 251.50 | 254.85 | 251.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 253.30 | 254.63 | 252.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:30:00 | 253.40 | 254.63 | 252.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 252.60 | 254.56 | 252.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 250.50 | 254.56 | 252.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 253.10 | 254.27 | 252.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 252.65 | 254.27 | 252.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 251.90 | 253.79 | 252.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:45:00 | 251.70 | 253.79 | 252.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 254.25 | 253.89 | 252.96 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 15:15:00 | 250.75 | 252.60 | 252.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 249.50 | 251.61 | 252.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 11:15:00 | 252.05 | 251.01 | 251.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 252.05 | 251.01 | 251.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 252.05 | 251.01 | 251.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 12:00:00 | 252.05 | 251.01 | 251.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 253.50 | 251.51 | 251.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 15:15:00 | 249.80 | 251.88 | 251.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 237.31 | 249.04 | 250.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 247.25 | 249.04 | 250.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 14:15:00 | 234.89 | 241.24 | 245.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-13 10:15:00 | 241.35 | 239.86 | 243.85 | SL hit (close>ema200) qty=0.50 sl=239.86 alert=retest2 |

### Cycle 38 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 253.60 | 246.05 | 245.33 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 244.90 | 247.84 | 247.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 243.50 | 246.10 | 247.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 235.40 | 234.05 | 237.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 11:00:00 | 235.40 | 234.05 | 237.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 232.65 | 232.76 | 234.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 235.50 | 232.76 | 234.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 237.95 | 233.84 | 234.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 238.85 | 233.84 | 234.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 236.20 | 234.31 | 234.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 234.25 | 234.44 | 234.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 13:15:00 | 233.90 | 231.32 | 231.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 14:15:00 | 232.55 | 231.95 | 231.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 14:15:00 | 232.55 | 231.95 | 231.90 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 229.00 | 231.81 | 232.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 10:15:00 | 225.05 | 228.80 | 229.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 226.80 | 226.28 | 227.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 11:15:00 | 227.75 | 226.46 | 227.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 227.75 | 226.46 | 227.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 227.75 | 226.46 | 227.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 228.55 | 226.88 | 227.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 228.30 | 226.88 | 227.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 228.45 | 227.19 | 227.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:00:00 | 228.45 | 227.19 | 227.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 230.75 | 228.41 | 228.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 235.00 | 230.48 | 229.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 12:15:00 | 229.85 | 230.35 | 229.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 13:00:00 | 229.85 | 230.35 | 229.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 229.10 | 230.10 | 229.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 229.10 | 230.10 | 229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 228.70 | 229.82 | 229.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 15:00:00 | 228.70 | 229.82 | 229.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 229.05 | 229.67 | 229.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 229.75 | 229.67 | 229.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 11:30:00 | 229.90 | 229.59 | 229.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 13:15:00 | 229.55 | 229.54 | 229.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 15:15:00 | 228.80 | 229.17 | 229.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-10-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 15:15:00 | 228.80 | 229.17 | 229.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 225.75 | 228.49 | 228.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 12:15:00 | 222.05 | 221.41 | 223.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-17 12:30:00 | 222.10 | 221.41 | 223.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 221.20 | 221.37 | 223.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 15:00:00 | 220.80 | 222.62 | 223.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:30:00 | 220.25 | 222.19 | 222.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 14:15:00 | 225.50 | 223.30 | 223.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 14:15:00 | 225.50 | 223.30 | 223.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 15:15:00 | 226.55 | 223.95 | 223.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 224.10 | 224.92 | 224.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 11:15:00 | 224.10 | 224.92 | 224.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 224.10 | 224.92 | 224.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 12:00:00 | 224.10 | 224.92 | 224.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 224.85 | 224.90 | 224.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 13:30:00 | 225.65 | 224.78 | 224.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 14:15:00 | 225.80 | 224.78 | 224.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 15:00:00 | 226.30 | 225.09 | 224.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 217.70 | 223.69 | 223.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 217.70 | 223.69 | 223.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 10:15:00 | 216.95 | 222.34 | 223.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 207.80 | 207.21 | 210.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 208.20 | 208.12 | 210.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 208.20 | 208.12 | 210.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 10:15:00 | 207.90 | 208.12 | 210.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:00:00 | 207.70 | 208.13 | 209.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:30:00 | 207.70 | 208.12 | 209.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 09:30:00 | 207.40 | 208.24 | 209.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 208.55 | 208.31 | 209.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 10:30:00 | 209.45 | 208.31 | 209.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 209.70 | 208.58 | 209.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:45:00 | 210.30 | 208.58 | 209.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 210.00 | 208.87 | 209.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 12:30:00 | 210.25 | 208.87 | 209.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 209.95 | 209.49 | 209.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:15:00 | 211.90 | 209.49 | 209.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 209.95 | 209.58 | 209.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:30:00 | 212.50 | 209.58 | 209.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 209.10 | 209.48 | 209.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-31 13:15:00 | 210.40 | 209.66 | 209.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 13:15:00 | 210.40 | 209.66 | 209.62 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 15:15:00 | 208.45 | 209.43 | 209.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 11:15:00 | 207.95 | 209.00 | 209.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 205.35 | 204.77 | 206.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 09:45:00 | 204.50 | 204.77 | 206.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 203.95 | 204.61 | 206.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:30:00 | 206.65 | 204.61 | 206.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 206.30 | 204.80 | 206.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 206.30 | 204.80 | 206.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 206.40 | 205.12 | 206.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:45:00 | 205.85 | 205.12 | 206.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 207.55 | 205.61 | 206.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 207.55 | 205.61 | 206.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 209.20 | 206.32 | 206.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 209.20 | 206.32 | 206.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 10:15:00 | 208.35 | 207.16 | 207.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 215.30 | 209.04 | 207.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 213.70 | 213.94 | 212.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 12:45:00 | 213.55 | 213.94 | 212.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 214.15 | 214.22 | 213.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 214.00 | 214.22 | 213.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 213.55 | 214.09 | 213.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:45:00 | 213.50 | 214.09 | 213.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 213.30 | 213.93 | 213.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 13:30:00 | 214.80 | 213.74 | 213.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 14:15:00 | 212.65 | 213.52 | 213.40 | SL hit (close<static) qty=1.00 sl=213.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 211.50 | 213.00 | 213.18 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 214.05 | 213.28 | 213.19 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 212.50 | 213.16 | 213.18 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 214.50 | 213.37 | 213.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 12:15:00 | 219.35 | 215.64 | 214.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 14:15:00 | 217.95 | 218.93 | 217.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 14:15:00 | 217.95 | 218.93 | 217.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 217.95 | 218.93 | 217.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 217.95 | 218.93 | 217.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 217.40 | 218.63 | 217.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:15:00 | 216.70 | 218.63 | 217.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 218.10 | 218.52 | 217.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 12:45:00 | 219.30 | 218.98 | 217.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 219.15 | 218.77 | 218.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:15:00 | 219.65 | 218.73 | 218.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 12:15:00 | 218.90 | 218.72 | 218.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 218.80 | 218.74 | 218.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-22 12:15:00 | 215.65 | 217.88 | 218.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 215.65 | 217.88 | 218.11 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 219.80 | 218.04 | 217.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 10:15:00 | 221.00 | 218.63 | 218.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 13:15:00 | 253.85 | 254.46 | 249.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 13:45:00 | 253.80 | 254.46 | 249.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 251.70 | 253.62 | 250.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 10:30:00 | 251.00 | 253.62 | 250.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 249.55 | 252.81 | 250.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 246.80 | 252.81 | 250.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 249.00 | 252.05 | 250.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 15:00:00 | 256.20 | 252.38 | 250.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 250.60 | 252.70 | 252.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 11:45:00 | 250.75 | 251.91 | 251.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 14:15:00 | 265.95 | 268.74 | 269.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 265.95 | 268.74 | 269.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 09:15:00 | 259.20 | 266.41 | 267.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 11:15:00 | 263.35 | 261.90 | 263.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 11:15:00 | 263.35 | 261.90 | 263.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 263.35 | 261.90 | 263.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:45:00 | 263.95 | 261.90 | 263.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 263.15 | 262.15 | 263.86 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 269.95 | 265.65 | 265.11 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 261.50 | 264.54 | 264.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 252.95 | 262.22 | 263.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 252.00 | 251.48 | 255.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 253.50 | 251.48 | 255.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 254.75 | 252.13 | 255.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:45:00 | 255.15 | 252.13 | 255.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 253.80 | 252.46 | 255.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:30:00 | 254.65 | 252.46 | 255.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 253.60 | 253.05 | 255.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:00:00 | 253.60 | 253.05 | 255.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 251.90 | 252.82 | 254.78 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 13:15:00 | 258.60 | 255.35 | 255.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 260.55 | 256.39 | 255.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 260.00 | 260.19 | 257.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 260.00 | 260.19 | 257.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 256.75 | 259.50 | 257.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 259.35 | 259.50 | 257.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 257.75 | 259.15 | 257.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 257.25 | 259.15 | 257.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 257.75 | 258.87 | 257.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:45:00 | 261.80 | 258.47 | 258.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 261.75 | 258.88 | 258.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 261.15 | 261.02 | 260.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:15:00 | 261.10 | 259.70 | 259.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 261.15 | 259.99 | 259.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:15:00 | 265.60 | 260.15 | 259.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:45:00 | 263.55 | 260.67 | 260.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 09:15:00 | 260.25 | 265.86 | 265.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 260.25 | 265.86 | 265.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 257.75 | 261.38 | 263.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 261.15 | 260.40 | 262.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 12:15:00 | 261.15 | 260.40 | 262.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 261.15 | 260.40 | 262.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 14:15:00 | 258.80 | 260.40 | 261.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 14:15:00 | 259.95 | 259.05 | 260.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 09:30:00 | 259.95 | 258.89 | 259.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 12:15:00 | 261.70 | 259.83 | 259.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 12:15:00 | 261.70 | 259.83 | 259.61 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 15:15:00 | 258.40 | 259.47 | 259.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 255.10 | 258.60 | 259.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 258.75 | 257.71 | 258.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 258.75 | 257.71 | 258.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 258.75 | 257.71 | 258.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:45:00 | 258.85 | 257.71 | 258.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 260.95 | 258.36 | 258.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:45:00 | 261.30 | 258.36 | 258.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 258.80 | 258.45 | 258.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 12:45:00 | 258.05 | 257.79 | 258.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 15:15:00 | 256.50 | 258.04 | 258.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 245.15 | 252.69 | 254.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 243.67 | 252.69 | 254.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 253.75 | 250.94 | 252.68 | SL hit (close>ema200) qty=0.50 sl=250.94 alert=retest2 |

### Cycle 62 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 252.30 | 245.13 | 244.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 256.05 | 248.65 | 246.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 260.05 | 260.16 | 255.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 14:30:00 | 260.65 | 260.16 | 255.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 250.25 | 259.13 | 258.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 249.75 | 259.13 | 258.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 244.15 | 256.13 | 256.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 234.10 | 239.53 | 243.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 238.05 | 237.17 | 241.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 11:00:00 | 238.05 | 237.17 | 241.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 240.65 | 238.13 | 241.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 12:45:00 | 240.85 | 238.13 | 241.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 239.95 | 238.49 | 240.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 11:00:00 | 238.05 | 239.97 | 240.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 14:00:00 | 238.40 | 239.34 | 240.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 11:15:00 | 241.95 | 237.42 | 238.68 | SL hit (close>static) qty=1.00 sl=241.20 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 12:15:00 | 248.50 | 239.64 | 239.57 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 238.15 | 240.23 | 240.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 235.40 | 239.26 | 240.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 234.45 | 234.18 | 236.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 09:45:00 | 234.50 | 234.18 | 236.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 235.10 | 234.41 | 235.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:45:00 | 234.80 | 234.41 | 235.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 234.70 | 234.35 | 235.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:45:00 | 234.85 | 234.35 | 235.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 234.60 | 234.40 | 235.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 236.50 | 234.40 | 235.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 237.00 | 234.92 | 235.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:45:00 | 237.00 | 234.92 | 235.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 237.60 | 235.46 | 235.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 11:00:00 | 237.60 | 235.46 | 235.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 13:15:00 | 236.70 | 236.07 | 236.02 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 235.30 | 235.86 | 235.93 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 242.45 | 237.18 | 236.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 11:15:00 | 243.50 | 239.08 | 237.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 241.10 | 241.89 | 239.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 241.10 | 241.89 | 239.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 241.10 | 241.89 | 239.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 241.10 | 241.89 | 239.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 240.75 | 241.44 | 239.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:45:00 | 239.90 | 241.44 | 239.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 241.60 | 241.38 | 240.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 14:30:00 | 242.30 | 241.23 | 240.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 15:15:00 | 241.95 | 241.23 | 240.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:45:00 | 241.75 | 241.27 | 240.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:00:00 | 241.75 | 241.36 | 240.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 240.25 | 241.09 | 240.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 240.25 | 241.09 | 240.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 241.20 | 241.11 | 240.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:15:00 | 244.15 | 241.11 | 240.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 09:30:00 | 242.20 | 244.20 | 243.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 10:00:00 | 244.80 | 244.20 | 243.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 10:00:00 | 242.10 | 245.03 | 244.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 10:15:00 | 241.75 | 244.37 | 244.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 241.75 | 244.37 | 244.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 13:15:00 | 240.90 | 242.97 | 243.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 229.10 | 226.80 | 230.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 229.10 | 226.80 | 230.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 230.70 | 227.95 | 230.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 230.80 | 227.95 | 230.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 230.25 | 228.41 | 230.17 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 232.60 | 230.95 | 230.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 12:15:00 | 233.00 | 231.64 | 231.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 13:15:00 | 231.60 | 232.17 | 231.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 13:15:00 | 231.60 | 232.17 | 231.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 231.60 | 232.17 | 231.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:00:00 | 231.60 | 232.17 | 231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 232.10 | 232.15 | 231.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 231.70 | 232.15 | 231.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 232.75 | 232.27 | 231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:15:00 | 233.55 | 232.27 | 231.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 09:15:00 | 231.50 | 232.12 | 231.80 | SL hit (close<static) qty=1.00 sl=231.60 alert=retest2 |

### Cycle 71 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 229.85 | 231.50 | 231.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 13:15:00 | 228.85 | 230.67 | 231.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 225.35 | 224.51 | 226.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:30:00 | 224.75 | 224.51 | 226.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 225.45 | 224.69 | 226.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:00:00 | 223.45 | 225.18 | 226.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:45:00 | 222.80 | 224.62 | 225.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 14:15:00 | 212.28 | 216.16 | 219.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 15:15:00 | 211.66 | 215.30 | 219.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 12:15:00 | 201.10 | 208.60 | 214.42 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 202.20 | 200.65 | 200.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 204.15 | 201.35 | 200.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 210.80 | 211.72 | 209.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:15:00 | 213.45 | 211.72 | 209.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 15:15:00 | 213.40 | 213.09 | 211.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 213.40 | 213.15 | 211.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 216.15 | 213.15 | 211.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 14:15:00 | 224.12 | 221.17 | 218.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 14:15:00 | 224.07 | 221.17 | 218.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-03 13:15:00 | 223.40 | 224.08 | 221.26 | SL hit (close<ema200) qty=0.50 sl=224.08 alert=retest1 |

### Cycle 73 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 11:15:00 | 225.85 | 226.44 | 226.46 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 229.30 | 226.90 | 226.66 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 225.20 | 226.74 | 226.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 224.60 | 226.31 | 226.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 223.95 | 223.70 | 224.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:30:00 | 223.90 | 223.70 | 224.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 224.50 | 223.86 | 224.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 224.50 | 223.86 | 224.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 224.30 | 223.95 | 224.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 225.10 | 223.95 | 224.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 223.90 | 223.94 | 224.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:45:00 | 225.30 | 223.94 | 224.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 226.70 | 224.49 | 224.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 226.70 | 224.49 | 224.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 226.80 | 224.95 | 224.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:15:00 | 227.95 | 224.95 | 224.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 227.95 | 225.55 | 225.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 10:15:00 | 228.25 | 226.42 | 225.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 225.60 | 226.67 | 226.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 13:15:00 | 225.60 | 226.67 | 226.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 225.60 | 226.67 | 226.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:00:00 | 225.60 | 226.67 | 226.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 221.05 | 225.55 | 225.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 220.30 | 223.47 | 224.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 225.15 | 223.63 | 224.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 12:15:00 | 225.15 | 223.63 | 224.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 225.15 | 223.63 | 224.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:00:00 | 225.15 | 223.63 | 224.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 224.85 | 223.87 | 224.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 224.85 | 223.87 | 224.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 223.95 | 223.89 | 224.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 224.05 | 223.89 | 224.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 224.60 | 224.03 | 224.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 226.00 | 224.03 | 224.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 222.45 | 223.71 | 224.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:15:00 | 220.85 | 222.52 | 223.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 14:00:00 | 220.80 | 222.18 | 223.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:30:00 | 220.15 | 221.43 | 222.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 11:15:00 | 221.00 | 221.43 | 222.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 224.10 | 221.96 | 222.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:00:00 | 224.10 | 221.96 | 222.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 224.45 | 222.46 | 222.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:45:00 | 224.65 | 222.46 | 222.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 225.55 | 223.08 | 223.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 13:15:00 | 225.55 | 223.08 | 223.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 228.50 | 224.49 | 223.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 227.10 | 227.30 | 225.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 227.00 | 227.30 | 225.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 227.00 | 227.24 | 225.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 228.15 | 226.52 | 226.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:00:00 | 228.40 | 226.94 | 226.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 228.95 | 227.91 | 227.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:00:00 | 228.25 | 227.58 | 227.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 226.80 | 227.48 | 227.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:00:00 | 226.80 | 227.48 | 227.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 226.60 | 227.30 | 227.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:45:00 | 227.15 | 227.30 | 227.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 226.20 | 227.08 | 227.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 227.30 | 227.08 | 227.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 224.90 | 226.88 | 227.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 224.90 | 226.88 | 227.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 223.90 | 226.28 | 226.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 227.60 | 225.43 | 225.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 227.60 | 225.43 | 225.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 227.60 | 225.43 | 225.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:45:00 | 227.80 | 225.43 | 225.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 225.85 | 225.51 | 225.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 224.50 | 225.51 | 225.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:15:00 | 213.27 | 216.32 | 219.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 210.90 | 210.57 | 214.70 | SL hit (close>ema200) qty=0.50 sl=210.57 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 209.45 | 206.85 | 206.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 210.60 | 207.60 | 207.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 209.50 | 209.64 | 208.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:15:00 | 209.35 | 209.64 | 208.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 209.60 | 209.63 | 208.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 209.45 | 209.63 | 208.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 208.55 | 209.42 | 208.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 208.55 | 209.42 | 208.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 208.60 | 209.25 | 208.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 209.25 | 209.25 | 208.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:30:00 | 209.50 | 209.11 | 208.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:30:00 | 209.35 | 209.00 | 208.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 207.35 | 208.67 | 208.62 | SL hit (close<static) qty=1.00 sl=208.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 209.75 | 210.56 | 210.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 208.50 | 209.66 | 210.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 209.60 | 209.35 | 209.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 209.60 | 209.35 | 209.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 209.60 | 209.35 | 209.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 210.40 | 209.35 | 209.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 208.90 | 209.25 | 209.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 209.80 | 209.25 | 209.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 209.55 | 209.29 | 209.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:45:00 | 209.80 | 209.29 | 209.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 209.30 | 209.29 | 209.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:00:00 | 208.40 | 209.08 | 209.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 13:15:00 | 210.25 | 209.50 | 209.55 | SL hit (close>static) qty=1.00 sl=210.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 09:15:00 | 212.75 | 209.93 | 209.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 15:15:00 | 213.90 | 211.97 | 210.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 12:15:00 | 213.10 | 213.25 | 211.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 213.10 | 213.25 | 211.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 213.10 | 213.25 | 211.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 212.20 | 213.25 | 211.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 213.40 | 213.13 | 212.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 215.60 | 213.13 | 212.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 218.35 | 214.18 | 212.79 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 209.60 | 212.36 | 212.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 206.35 | 210.21 | 211.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 206.50 | 205.87 | 208.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 206.50 | 205.87 | 208.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 212.35 | 206.84 | 208.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:45:00 | 213.05 | 206.84 | 208.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 212.65 | 208.00 | 208.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:00:00 | 212.65 | 208.00 | 208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 215.00 | 209.40 | 209.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 217.75 | 211.07 | 209.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 205.85 | 211.81 | 210.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 205.85 | 211.81 | 210.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 205.85 | 211.81 | 210.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 203.20 | 211.81 | 210.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 195.00 | 208.45 | 209.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 175.50 | 201.86 | 206.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 193.45 | 191.58 | 197.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:15:00 | 193.70 | 191.58 | 197.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 196.85 | 193.09 | 197.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 196.85 | 193.09 | 197.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 196.00 | 193.68 | 197.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 197.10 | 193.68 | 197.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 205.90 | 196.62 | 197.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 206.20 | 196.62 | 197.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 207.55 | 198.81 | 198.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 214.35 | 205.87 | 202.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 217.99 | 218.18 | 213.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 217.99 | 218.18 | 213.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 220.05 | 219.34 | 217.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 221.40 | 219.20 | 218.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:45:00 | 220.50 | 220.99 | 220.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 220.60 | 220.57 | 220.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 12:00:00 | 220.87 | 220.30 | 220.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 219.79 | 220.20 | 220.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 219.79 | 220.20 | 220.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 219.82 | 220.12 | 220.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 219.82 | 220.12 | 220.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 217.56 | 219.61 | 219.91 | Break + close below crossover candle low |

### Cycle 88 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 224.45 | 220.46 | 220.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 228.30 | 222.03 | 220.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 230.98 | 231.70 | 228.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 230.98 | 231.70 | 228.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 230.74 | 231.31 | 229.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 225.20 | 231.31 | 229.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 228.20 | 230.69 | 228.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 228.20 | 230.69 | 228.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 235.65 | 231.68 | 229.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 12:30:00 | 236.50 | 232.52 | 230.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 242.77 | 231.13 | 231.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-26 09:15:00 | 260.15 | 236.56 | 233.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 281.50 | 283.85 | 283.87 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 15:15:00 | 286.35 | 284.29 | 284.05 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 280.50 | 283.96 | 284.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 277.85 | 282.74 | 283.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 11:15:00 | 279.10 | 278.71 | 280.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 11:15:00 | 279.10 | 278.71 | 280.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 279.10 | 278.71 | 280.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 276.65 | 280.30 | 280.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 284.75 | 281.16 | 281.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 284.75 | 281.16 | 281.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 294.25 | 285.38 | 283.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 300.25 | 301.69 | 296.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 300.25 | 301.69 | 296.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 300.25 | 301.69 | 296.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 297.50 | 301.69 | 296.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 297.50 | 300.07 | 297.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 302.10 | 300.07 | 297.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 302.25 | 300.51 | 297.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 306.35 | 301.30 | 298.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 09:15:00 | 336.99 | 322.22 | 313.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 366.00 | 368.02 | 368.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 364.35 | 367.29 | 367.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 364.10 | 363.45 | 365.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-01 15:15:00 | 363.40 | 363.45 | 365.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 367.95 | 364.34 | 365.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 367.95 | 364.34 | 365.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 372.05 | 365.88 | 365.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:30:00 | 372.15 | 365.88 | 365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 370.80 | 366.87 | 366.33 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 364.75 | 367.20 | 367.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 363.20 | 366.40 | 367.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 366.40 | 366.03 | 366.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 366.40 | 366.03 | 366.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 366.40 | 366.03 | 366.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 366.00 | 366.03 | 366.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 369.25 | 366.67 | 367.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 369.25 | 366.67 | 367.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 366.50 | 366.64 | 366.98 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 367.85 | 367.27 | 367.20 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 10:15:00 | 366.35 | 367.47 | 367.52 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 368.50 | 367.67 | 367.61 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 367.25 | 367.52 | 367.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 366.25 | 367.10 | 367.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 367.50 | 366.87 | 367.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 367.50 | 366.87 | 367.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 367.50 | 366.87 | 367.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 367.30 | 366.87 | 367.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 366.85 | 366.86 | 367.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 366.85 | 366.86 | 367.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 366.65 | 366.82 | 367.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 366.40 | 366.82 | 367.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:45:00 | 365.80 | 366.63 | 366.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 365.65 | 366.33 | 366.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:45:00 | 366.00 | 365.33 | 365.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 365.00 | 365.26 | 365.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 10:15:00 | 363.95 | 364.97 | 365.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 11:45:00 | 364.00 | 364.62 | 364.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 12:45:00 | 364.00 | 364.53 | 364.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:15:00 | 363.85 | 364.58 | 364.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 366.25 | 364.61 | 364.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 367.30 | 364.61 | 364.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-22 10:15:00 | 366.55 | 365.00 | 364.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 366.55 | 365.00 | 364.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 370.50 | 366.10 | 365.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 366.70 | 367.35 | 366.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 366.70 | 367.35 | 366.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 366.70 | 367.35 | 366.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 366.70 | 367.35 | 366.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 367.75 | 367.43 | 366.51 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 366.00 | 366.69 | 366.71 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 367.75 | 366.89 | 366.79 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 366.00 | 366.76 | 366.80 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 367.55 | 366.92 | 366.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 372.55 | 368.29 | 367.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 369.00 | 369.51 | 368.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 369.00 | 369.51 | 368.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 369.00 | 369.51 | 368.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 368.40 | 369.51 | 368.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 366.85 | 368.98 | 368.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 366.85 | 368.98 | 368.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 363.95 | 367.97 | 367.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:00:00 | 363.95 | 367.97 | 367.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 364.70 | 367.32 | 367.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 14:15:00 | 363.15 | 366.48 | 367.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 367.80 | 366.02 | 366.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 367.80 | 366.02 | 366.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 367.80 | 366.02 | 366.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 368.90 | 366.02 | 366.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 366.85 | 366.18 | 366.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:15:00 | 365.05 | 366.31 | 366.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:30:00 | 365.15 | 364.74 | 365.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 14:00:00 | 365.10 | 364.74 | 365.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 365.30 | 364.90 | 365.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 364.75 | 364.87 | 365.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:30:00 | 365.75 | 364.87 | 365.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 365.55 | 364.94 | 365.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 365.55 | 364.94 | 365.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 364.75 | 364.90 | 365.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 364.35 | 365.35 | 365.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:00:00 | 364.40 | 365.05 | 365.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 13:30:00 | 364.45 | 364.99 | 365.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 364.55 | 364.96 | 365.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 365.50 | 365.07 | 365.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 366.95 | 365.07 | 365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 365.00 | 365.05 | 365.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 363.55 | 364.59 | 364.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:00:00 | 363.45 | 364.15 | 364.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:30:00 | 363.95 | 363.98 | 364.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:45:00 | 363.05 | 363.62 | 364.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 365.80 | 363.73 | 363.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 365.70 | 363.73 | 363.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-09 15:15:00 | 366.20 | 364.22 | 364.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 366.20 | 364.22 | 364.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 367.15 | 365.48 | 364.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 14:15:00 | 364.50 | 365.81 | 365.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 364.50 | 365.81 | 365.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 364.50 | 365.81 | 365.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:30:00 | 364.40 | 365.81 | 365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 365.20 | 365.69 | 365.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 364.75 | 365.69 | 365.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 365.20 | 365.59 | 365.17 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 363.35 | 364.76 | 364.91 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 367.00 | 364.57 | 364.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 378.70 | 367.63 | 365.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 370.25 | 371.10 | 368.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 370.25 | 371.10 | 368.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 370.25 | 371.10 | 368.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 367.15 | 371.10 | 368.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 369.55 | 371.48 | 369.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:45:00 | 369.00 | 371.48 | 369.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 368.85 | 370.96 | 369.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 368.35 | 370.96 | 369.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 369.00 | 370.57 | 369.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 368.00 | 370.57 | 369.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 368.65 | 370.91 | 370.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 368.65 | 370.91 | 370.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 368.60 | 370.45 | 370.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 369.30 | 370.45 | 370.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 366.70 | 369.42 | 369.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 365.25 | 368.14 | 369.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 365.45 | 365.00 | 366.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 365.45 | 365.00 | 366.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 366.75 | 365.35 | 366.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 366.75 | 365.35 | 366.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 366.55 | 365.59 | 366.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:30:00 | 366.55 | 365.59 | 366.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 367.25 | 365.92 | 366.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:00:00 | 367.25 | 365.92 | 366.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 366.80 | 366.10 | 366.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 14:30:00 | 365.70 | 366.10 | 366.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 365.25 | 366.28 | 366.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 10:00:00 | 365.25 | 366.07 | 366.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 365.20 | 364.73 | 365.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 361.85 | 362.75 | 363.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 359.60 | 361.30 | 362.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 360.20 | 361.74 | 362.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 365.90 | 363.02 | 362.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 365.90 | 363.02 | 362.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 15:15:00 | 367.00 | 363.82 | 363.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 364.30 | 364.55 | 364.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 364.30 | 364.55 | 364.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 364.30 | 364.55 | 364.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 364.30 | 364.55 | 364.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 364.10 | 364.46 | 364.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 14:45:00 | 366.20 | 364.80 | 364.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:45:00 | 364.95 | 364.63 | 364.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 363.05 | 364.16 | 364.12 | SL hit (close<static) qty=1.00 sl=363.40 alert=retest2 |

### Cycle 111 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 362.40 | 363.81 | 363.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 361.00 | 362.98 | 363.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 13:15:00 | 362.65 | 362.52 | 363.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:00:00 | 362.65 | 362.52 | 363.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 361.70 | 362.36 | 362.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 14:45:00 | 364.00 | 362.36 | 362.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 362.50 | 362.38 | 362.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 360.60 | 362.38 | 362.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:00:00 | 361.30 | 361.09 | 361.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:00:00 | 361.30 | 361.13 | 361.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 363.55 | 362.12 | 361.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 363.55 | 362.12 | 361.99 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 362.00 | 362.42 | 362.45 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 362.75 | 362.52 | 362.49 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 362.10 | 362.59 | 362.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 360.40 | 361.80 | 362.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 362.00 | 361.71 | 362.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 362.00 | 361.71 | 362.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 362.00 | 361.71 | 362.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 362.05 | 361.71 | 362.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 362.00 | 361.77 | 362.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 361.60 | 361.81 | 362.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 360.40 | 356.55 | 356.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 360.40 | 356.55 | 356.55 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 13:15:00 | 356.00 | 356.74 | 356.79 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 358.10 | 357.07 | 356.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 360.00 | 358.15 | 357.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 362.10 | 362.13 | 360.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 362.10 | 362.13 | 360.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 360.40 | 361.78 | 360.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:30:00 | 359.25 | 361.78 | 360.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 359.45 | 361.32 | 360.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 359.45 | 361.32 | 360.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 359.50 | 360.95 | 360.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:45:00 | 359.90 | 360.95 | 360.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 361.20 | 360.91 | 360.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:30:00 | 360.05 | 360.91 | 360.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 360.70 | 360.86 | 360.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:15:00 | 360.20 | 360.86 | 360.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 360.20 | 360.73 | 360.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 361.25 | 360.73 | 360.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 362.00 | 360.99 | 360.51 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 360.25 | 360.93 | 361.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 359.50 | 360.54 | 360.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 358.80 | 358.27 | 359.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 14:15:00 | 358.80 | 358.27 | 359.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 358.80 | 358.27 | 359.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 358.80 | 358.27 | 359.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 362.90 | 359.20 | 359.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 357.00 | 359.20 | 359.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:30:00 | 357.95 | 358.02 | 358.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 357.55 | 358.16 | 358.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 358.00 | 358.15 | 358.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 358.00 | 358.12 | 358.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 358.00 | 358.12 | 358.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 357.65 | 358.03 | 358.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 357.65 | 358.03 | 358.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 355.55 | 356.10 | 357.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 15:00:00 | 355.55 | 356.10 | 357.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 352.30 | 354.87 | 356.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-18 15:15:00 | 357.10 | 355.60 | 355.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 15:15:00 | 357.10 | 355.60 | 355.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 357.30 | 356.20 | 355.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 11:15:00 | 356.05 | 356.17 | 355.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 11:15:00 | 356.05 | 356.17 | 355.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 356.05 | 356.17 | 355.88 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 355.05 | 355.65 | 355.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 354.60 | 355.28 | 355.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 355.90 | 355.41 | 355.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 355.90 | 355.41 | 355.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 355.90 | 355.41 | 355.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 355.90 | 355.41 | 355.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 356.00 | 355.52 | 355.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:15:00 | 355.75 | 355.52 | 355.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 356.60 | 355.74 | 355.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 358.85 | 356.49 | 356.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 358.50 | 358.92 | 358.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 359.20 | 358.92 | 358.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 359.85 | 359.10 | 358.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:30:00 | 361.50 | 359.81 | 358.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 361.20 | 360.65 | 359.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 366.30 | 368.03 | 368.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 366.30 | 368.03 | 368.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 360.20 | 366.46 | 367.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 346.15 | 345.11 | 349.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 346.15 | 345.11 | 349.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 346.15 | 345.11 | 349.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 344.05 | 344.87 | 347.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 342.10 | 342.74 | 343.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 09:15:00 | 341.20 | 337.89 | 337.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 09:15:00 | 341.20 | 337.89 | 337.72 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 334.05 | 337.49 | 337.95 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 09:15:00 | 365.75 | 343.38 | 340.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 376.45 | 372.99 | 367.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 373.35 | 373.53 | 368.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 373.35 | 373.53 | 368.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 377.05 | 377.84 | 376.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 375.10 | 377.84 | 376.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 376.25 | 377.52 | 376.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 379.35 | 377.52 | 376.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 15:00:00 | 377.50 | 377.66 | 376.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 377.50 | 377.66 | 377.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 11:30:00 | 377.60 | 377.30 | 377.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 377.10 | 377.31 | 377.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 377.10 | 377.31 | 377.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 378.15 | 377.48 | 377.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 377.25 | 377.48 | 377.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 378.65 | 377.79 | 377.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 376.40 | 377.20 | 377.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 376.40 | 377.20 | 377.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 375.20 | 376.55 | 376.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 377.00 | 376.57 | 376.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 377.00 | 376.57 | 376.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 377.00 | 376.57 | 376.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 377.00 | 376.57 | 376.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 377.00 | 376.66 | 376.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 377.25 | 376.66 | 376.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 377.10 | 376.75 | 376.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 377.35 | 376.75 | 376.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 377.55 | 376.91 | 376.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 377.55 | 376.91 | 376.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 377.70 | 377.07 | 377.03 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 376.75 | 377.11 | 377.16 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-13 11:15:00 | 377.60 | 377.08 | 377.06 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 15:15:00 | 376.80 | 377.06 | 377.07 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 09:15:00 | 377.30 | 377.11 | 377.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 379.50 | 377.84 | 377.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 13:15:00 | 378.30 | 378.33 | 377.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 13:45:00 | 378.65 | 378.33 | 377.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 378.00 | 378.27 | 377.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:45:00 | 377.80 | 378.27 | 377.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 379.30 | 378.47 | 378.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 377.85 | 378.41 | 378.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 378.60 | 378.45 | 378.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 378.60 | 378.45 | 378.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 379.05 | 378.57 | 378.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 379.60 | 378.71 | 378.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 377.55 | 378.57 | 378.36 | SL hit (close<static) qty=1.00 sl=378.10 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 376.00 | 378.05 | 378.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 11:15:00 | 373.65 | 377.17 | 377.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 377.35 | 377.00 | 377.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 14:00:00 | 377.35 | 377.00 | 377.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 378.80 | 377.36 | 377.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 378.50 | 377.36 | 377.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 380.15 | 377.92 | 377.89 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 368.25 | 375.98 | 377.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 355.85 | 371.96 | 375.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 15:15:00 | 272.10 | 270.27 | 281.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:15:00 | 280.50 | 270.27 | 281.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 281.00 | 272.42 | 281.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 281.00 | 272.42 | 281.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 281.00 | 274.13 | 281.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 281.30 | 274.13 | 281.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 281.50 | 275.61 | 281.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 283.10 | 275.61 | 281.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 279.00 | 276.29 | 280.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 278.40 | 276.52 | 280.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 271.75 | 277.21 | 280.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 09:15:00 | 264.48 | 271.19 | 275.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 275.80 | 267.97 | 270.84 | SL hit (close>ema200) qty=0.50 sl=267.97 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 280.20 | 270.61 | 269.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 281.80 | 274.49 | 271.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 280.00 | 280.11 | 277.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:00:00 | 280.00 | 280.11 | 277.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 282.00 | 280.92 | 278.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 285.60 | 281.09 | 278.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 274.35 | 279.08 | 278.87 | SL hit (close<static) qty=1.00 sl=277.85 alert=retest2 |

### Cycle 137 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 276.05 | 278.47 | 278.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 271.20 | 275.40 | 276.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 275.30 | 275.16 | 276.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 275.30 | 275.16 | 276.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 275.30 | 275.16 | 276.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 275.65 | 275.16 | 276.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 276.50 | 275.43 | 276.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 276.50 | 275.43 | 276.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 276.30 | 275.60 | 276.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 277.05 | 275.60 | 276.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 280.20 | 276.52 | 276.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 281.50 | 277.75 | 277.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 275.75 | 279.64 | 278.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 275.75 | 279.64 | 278.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 275.75 | 279.64 | 278.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 274.50 | 279.64 | 278.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 274.50 | 278.61 | 278.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 274.50 | 278.61 | 278.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 274.85 | 277.57 | 277.92 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 282.50 | 278.47 | 278.26 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 272.05 | 277.83 | 278.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 13:15:00 | 270.00 | 273.55 | 275.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 274.05 | 272.31 | 274.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 274.05 | 272.31 | 274.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 274.05 | 272.31 | 274.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 274.40 | 272.31 | 274.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 271.45 | 272.14 | 273.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:15:00 | 268.80 | 271.96 | 273.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:45:00 | 269.10 | 271.55 | 273.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 268.60 | 271.55 | 273.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 15:00:00 | 266.60 | 270.56 | 272.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 260.35 | 262.70 | 266.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:15:00 | 259.00 | 262.70 | 266.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:30:00 | 259.20 | 261.25 | 264.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 15:15:00 | 258.80 | 261.10 | 264.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 255.36 | 259.28 | 262.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 255.65 | 259.28 | 262.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 255.17 | 259.28 | 262.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 253.27 | 259.28 | 262.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 15:15:00 | 255.30 | 253.35 | 256.24 | SL hit (close>ema200) qty=0.50 sl=253.35 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 11:15:00 | 255.70 | 250.60 | 250.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 12:15:00 | 258.90 | 252.26 | 250.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 293.35 | 293.66 | 284.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 12:00:00 | 293.35 | 293.66 | 284.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 292.50 | 295.16 | 291.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 292.50 | 295.16 | 291.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 290.30 | 294.19 | 291.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 290.30 | 294.19 | 291.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 289.15 | 293.18 | 291.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 289.15 | 293.18 | 291.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 285.15 | 289.75 | 290.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 281.70 | 288.14 | 289.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 281.00 | 280.94 | 283.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 09:15:00 | 276.20 | 280.94 | 283.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 275.00 | 270.01 | 271.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 275.00 | 270.01 | 271.76 | SL hit (close>ema400) qty=1.00 sl=271.76 alert=retest1 |

### Cycle 144 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 273.35 | 272.33 | 272.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 277.10 | 274.51 | 273.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 278.10 | 278.28 | 276.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:15:00 | 288.10 | 278.28 | 276.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 282.00 | 282.80 | 280.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 278.40 | 281.92 | 279.99 | SL hit (close<ema400) qty=1.00 sl=279.99 alert=retest1 |

### Cycle 145 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 276.35 | 278.55 | 278.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 273.65 | 277.08 | 278.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 272.85 | 270.40 | 272.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 272.85 | 270.40 | 272.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 272.85 | 270.40 | 272.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 272.90 | 270.40 | 272.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 270.60 | 270.44 | 272.74 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 274.50 | 273.64 | 273.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 14:15:00 | 280.35 | 277.61 | 276.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.95 | 277.50 | 276.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 275.95 | 277.50 | 276.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 275.95 | 277.50 | 276.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 279.80 | 278.14 | 276.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:15:00 | 279.95 | 278.14 | 277.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 15:15:00 | 282.80 | 278.31 | 277.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 276.60 | 278.78 | 278.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 276.60 | 278.78 | 278.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 274.30 | 277.89 | 278.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 278.45 | 278.00 | 278.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 14:15:00 | 278.45 | 278.00 | 278.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 278.45 | 278.00 | 278.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 278.45 | 278.00 | 278.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 276.10 | 277.62 | 278.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 265.30 | 277.62 | 278.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 09:30:00 | 274.50 | 272.36 | 273.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 280.25 | 274.64 | 273.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 280.25 | 274.64 | 273.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 284.20 | 279.25 | 276.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 283.20 | 285.16 | 282.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 13:15:00 | 283.20 | 285.16 | 282.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 283.20 | 285.16 | 282.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:00:00 | 283.20 | 285.16 | 282.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 282.85 | 284.70 | 282.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:30:00 | 281.75 | 284.70 | 282.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 283.10 | 284.38 | 282.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 302.70 | 284.38 | 282.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 285.00 | 286.66 | 286.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 285.00 | 286.66 | 286.72 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 288.05 | 286.90 | 286.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 288.65 | 287.44 | 287.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 288.40 | 289.58 | 288.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 288.40 | 289.58 | 288.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 288.40 | 289.58 | 288.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 287.00 | 289.58 | 288.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 285.75 | 288.82 | 288.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 285.75 | 288.82 | 288.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 286.25 | 288.30 | 288.26 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 287.00 | 288.04 | 288.15 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 290.05 | 287.96 | 287.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 290.75 | 288.87 | 288.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 304.60 | 304.74 | 298.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 10:30:00 | 305.60 | 304.74 | 298.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 305.00 | 307.30 | 304.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 306.15 | 307.30 | 304.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 10:15:00 | 305.55 | 310.99 | 311.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 305.55 | 310.99 | 311.73 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 317.90 | 311.61 | 311.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 324.00 | 319.00 | 316.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 322.80 | 322.81 | 320.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 13:30:00 | 323.45 | 322.81 | 320.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 324.05 | 323.25 | 321.22 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 15:15:00 | 319.00 | 320.55 | 320.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 317.65 | 319.97 | 320.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 11:15:00 | 323.50 | 320.52 | 320.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 323.50 | 320.52 | 320.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 323.50 | 320.52 | 320.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 323.50 | 320.52 | 320.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 322.00 | 320.81 | 320.67 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 319.65 | 321.02 | 321.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 318.30 | 320.48 | 320.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 323.25 | 319.88 | 320.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 323.25 | 319.88 | 320.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 323.25 | 319.88 | 320.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 324.20 | 319.88 | 320.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 322.75 | 320.45 | 320.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 324.25 | 321.79 | 321.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 329.40 | 330.64 | 327.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 329.40 | 330.64 | 327.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 328.00 | 330.11 | 327.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 328.00 | 330.11 | 327.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 326.70 | 329.43 | 327.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 326.45 | 329.43 | 327.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 325.25 | 328.60 | 327.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 325.25 | 328.60 | 327.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 326.10 | 327.88 | 327.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 326.20 | 327.88 | 327.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 326.00 | 327.50 | 327.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 325.05 | 327.50 | 327.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 328.85 | 327.77 | 327.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 330.40 | 328.52 | 327.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 330.00 | 333.65 | 333.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 330.00 | 333.65 | 333.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 317.85 | 330.49 | 332.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 328.40 | 326.85 | 328.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 328.40 | 326.85 | 328.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 328.40 | 326.85 | 328.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 328.40 | 326.85 | 328.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 329.00 | 327.28 | 328.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 336.40 | 327.28 | 328.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 339.00 | 329.63 | 329.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:45:00 | 339.30 | 329.63 | 329.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 335.90 | 330.88 | 330.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 342.90 | 333.28 | 331.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 339.95 | 341.22 | 336.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:30:00 | 338.10 | 341.22 | 336.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 339.25 | 340.79 | 338.73 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 334.25 | 337.89 | 337.98 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 340.00 | 337.99 | 337.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 343.00 | 340.31 | 339.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 11:15:00 | 340.70 | 340.77 | 339.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 12:00:00 | 340.70 | 340.77 | 339.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 340.10 | 340.64 | 339.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 340.05 | 340.64 | 339.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 341.05 | 340.72 | 339.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:45:00 | 342.50 | 341.10 | 340.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:00:00 | 344.00 | 341.90 | 340.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:00:00 | 346.45 | 342.91 | 341.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 340.00 | 342.41 | 342.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 340.00 | 342.41 | 342.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 336.55 | 340.77 | 341.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 331.90 | 329.99 | 333.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 331.90 | 329.99 | 333.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 331.90 | 329.99 | 333.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 325.55 | 328.37 | 330.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 325.40 | 327.63 | 329.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 309.27 | 313.08 | 318.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 309.13 | 313.08 | 318.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 311.70 | 311.13 | 315.56 | SL hit (close>ema200) qty=0.50 sl=311.13 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 325.10 | 315.26 | 315.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 326.90 | 317.59 | 316.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 336.25 | 340.86 | 335.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 336.25 | 340.86 | 335.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 336.25 | 340.86 | 335.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 336.25 | 340.86 | 335.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 336.15 | 339.92 | 335.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 334.05 | 339.92 | 335.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 334.60 | 338.85 | 335.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 334.60 | 338.85 | 335.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 336.40 | 338.36 | 335.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 339.00 | 338.36 | 335.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 344.00 | 344.29 | 344.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 344.00 | 344.29 | 344.31 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 344.55 | 344.34 | 344.33 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 343.90 | 344.25 | 344.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 339.75 | 343.35 | 343.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 342.75 | 342.33 | 343.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 13:00:00 | 342.75 | 342.33 | 343.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 343.55 | 342.57 | 343.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 343.55 | 342.57 | 343.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 345.70 | 343.20 | 343.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 345.70 | 343.20 | 343.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 345.60 | 343.68 | 343.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 349.00 | 345.34 | 344.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 347.35 | 347.61 | 346.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 347.35 | 347.61 | 346.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 347.35 | 347.50 | 346.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 346.90 | 347.50 | 346.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 345.30 | 347.31 | 346.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 345.30 | 347.31 | 346.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 344.75 | 346.80 | 346.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:15:00 | 344.10 | 346.80 | 346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 346.30 | 346.70 | 346.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 346.30 | 346.70 | 346.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 346.00 | 346.56 | 346.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 345.80 | 346.56 | 346.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 347.40 | 346.73 | 346.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 346.25 | 346.73 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 345.35 | 346.43 | 346.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 345.70 | 346.43 | 346.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 348.55 | 346.86 | 346.55 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 346.05 | 346.35 | 346.36 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 350.90 | 347.25 | 346.76 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 345.15 | 346.36 | 346.47 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 349.10 | 347.00 | 346.72 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 344.60 | 346.82 | 346.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 344.00 | 345.90 | 346.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 348.95 | 345.92 | 346.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 348.95 | 345.92 | 346.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 348.95 | 345.92 | 346.16 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 354.40 | 347.62 | 346.91 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 346.60 | 347.83 | 347.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 331.00 | 344.46 | 346.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 343.00 | 341.45 | 344.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 14:00:00 | 343.00 | 341.45 | 344.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 342.00 | 341.66 | 343.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 358.85 | 341.66 | 343.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 346.90 | 342.70 | 344.01 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 355.10 | 345.18 | 345.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 363.35 | 348.82 | 346.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 360.65 | 362.52 | 358.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 360.65 | 362.52 | 358.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 361.00 | 362.45 | 359.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 360.90 | 362.45 | 359.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 359.30 | 361.82 | 359.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 355.15 | 361.82 | 359.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 356.00 | 360.66 | 359.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:45:00 | 358.65 | 359.18 | 358.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 359.90 | 359.37 | 359.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 15:15:00 | 357.75 | 358.96 | 358.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 357.75 | 358.96 | 358.98 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 365.30 | 360.23 | 359.55 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 355.85 | 359.07 | 359.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 354.90 | 357.90 | 358.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 358.15 | 356.93 | 358.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 358.15 | 356.93 | 358.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 358.15 | 356.93 | 358.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 358.15 | 356.93 | 358.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 362.15 | 357.97 | 358.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 362.15 | 357.97 | 358.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 363.25 | 359.03 | 358.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 365.90 | 360.40 | 359.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 368.45 | 369.04 | 365.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 368.45 | 369.04 | 365.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 371.70 | 369.18 | 366.56 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 362.00 | 365.71 | 366.12 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 371.60 | 365.36 | 365.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 373.10 | 368.39 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 368.05 | 369.05 | 367.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 368.05 | 369.05 | 367.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 368.05 | 369.05 | 367.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 367.50 | 369.05 | 367.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 366.60 | 368.58 | 367.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 367.00 | 368.58 | 367.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 367.15 | 368.29 | 367.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 367.15 | 368.29 | 367.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 368.90 | 368.42 | 367.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:15:00 | 366.95 | 368.42 | 367.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 366.95 | 368.12 | 367.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 364.70 | 368.12 | 367.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 362.50 | 367.00 | 367.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 361.45 | 365.89 | 366.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 356.20 | 356.07 | 358.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:15:00 | 357.75 | 356.07 | 358.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 362.95 | 357.64 | 358.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 362.95 | 357.64 | 358.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 357.95 | 357.70 | 358.60 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 365.15 | 360.24 | 359.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 369.65 | 364.51 | 362.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 366.60 | 367.56 | 365.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 366.60 | 367.56 | 365.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 366.60 | 367.56 | 365.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 367.95 | 367.56 | 365.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 364.55 | 366.96 | 365.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 364.55 | 366.96 | 365.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 363.80 | 366.33 | 365.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 363.20 | 366.33 | 365.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 371.60 | 372.57 | 371.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:45:00 | 371.35 | 372.57 | 371.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 370.30 | 372.12 | 371.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 371.25 | 372.12 | 371.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 370.80 | 371.85 | 371.17 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 369.00 | 370.67 | 370.71 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 382.50 | 372.59 | 371.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 390.10 | 378.31 | 374.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 386.15 | 386.71 | 381.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:30:00 | 386.20 | 386.71 | 381.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 379.65 | 385.25 | 381.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 376.00 | 385.25 | 381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 378.95 | 383.99 | 381.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 378.95 | 383.99 | 381.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 372.25 | 379.29 | 379.67 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 381.90 | 379.75 | 379.68 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 377.15 | 379.53 | 379.61 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 381.95 | 379.46 | 379.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 14:15:00 | 382.75 | 380.12 | 379.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 12:15:00 | 383.10 | 383.57 | 382.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 383.10 | 383.57 | 382.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 383.90 | 383.64 | 382.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 387.45 | 383.79 | 382.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 384.25 | 384.18 | 383.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 384.00 | 383.95 | 383.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 11:45:00 | 384.00 | 383.77 | 383.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 385.10 | 384.04 | 383.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:30:00 | 385.35 | 384.04 | 383.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 394.55 | 398.21 | 395.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 394.55 | 398.21 | 395.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 395.70 | 397.71 | 395.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 396.65 | 397.71 | 395.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 405.45 | 396.98 | 396.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 393.95 | 398.68 | 398.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 393.95 | 398.68 | 398.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 392.70 | 397.48 | 398.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 391.15 | 390.77 | 393.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 391.15 | 390.77 | 393.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 391.15 | 390.77 | 393.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 391.15 | 390.77 | 393.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 392.80 | 391.19 | 392.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 392.80 | 391.19 | 392.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 394.15 | 391.78 | 392.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 394.15 | 391.78 | 392.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 394.40 | 392.30 | 392.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 397.10 | 392.30 | 392.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 391.15 | 391.62 | 392.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 392.00 | 391.62 | 392.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 393.80 | 392.05 | 392.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 394.65 | 392.05 | 392.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 395.50 | 392.74 | 392.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 398.55 | 394.33 | 393.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 397.25 | 397.42 | 395.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 397.25 | 397.42 | 395.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 402.00 | 398.41 | 396.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:30:00 | 396.80 | 398.41 | 396.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 402.20 | 403.10 | 401.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 401.95 | 403.10 | 401.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 399.70 | 402.42 | 401.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 399.70 | 402.42 | 401.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 399.00 | 401.74 | 401.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 396.85 | 401.74 | 401.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 397.50 | 400.89 | 400.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 391.95 | 397.49 | 398.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 385.40 | 384.42 | 388.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 385.40 | 384.42 | 388.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 386.70 | 383.85 | 386.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 387.50 | 383.85 | 386.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 384.70 | 384.02 | 386.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 386.10 | 384.02 | 386.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 388.90 | 384.99 | 386.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 388.90 | 384.99 | 386.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 388.00 | 385.60 | 386.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:00:00 | 386.60 | 386.10 | 386.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:30:00 | 386.90 | 386.27 | 386.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 386.45 | 386.10 | 386.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:00:00 | 384.65 | 384.06 | 385.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 384.15 | 384.08 | 385.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 386.30 | 384.08 | 385.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 386.35 | 384.53 | 385.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 388.00 | 384.53 | 385.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 386.00 | 384.83 | 385.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 383.80 | 384.54 | 385.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 386.85 | 384.40 | 384.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 386.85 | 384.40 | 384.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 390.60 | 385.64 | 384.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 389.35 | 389.52 | 387.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 389.35 | 389.52 | 387.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 385.40 | 388.75 | 387.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 391.35 | 388.75 | 387.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 395.45 | 390.09 | 388.18 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 387.00 | 390.66 | 391.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 385.95 | 389.72 | 390.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 391.00 | 389.45 | 390.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 391.00 | 389.45 | 390.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 391.00 | 389.45 | 390.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 391.45 | 389.45 | 390.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 391.00 | 389.76 | 390.32 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 392.90 | 390.94 | 390.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 396.45 | 392.44 | 391.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 390.70 | 392.48 | 391.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 390.70 | 392.48 | 391.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 390.70 | 392.48 | 391.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 390.70 | 392.48 | 391.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 392.30 | 392.45 | 391.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 393.20 | 392.45 | 391.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 392.65 | 393.10 | 392.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 389.30 | 391.95 | 392.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 389.30 | 391.95 | 392.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 387.45 | 391.05 | 391.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 388.15 | 387.90 | 389.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 388.15 | 387.90 | 389.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 388.70 | 388.06 | 389.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 389.05 | 388.06 | 389.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 390.70 | 388.59 | 389.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 390.70 | 388.59 | 389.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 392.85 | 389.44 | 389.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 392.85 | 389.44 | 389.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 393.15 | 390.18 | 390.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 395.65 | 392.43 | 391.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 392.30 | 392.86 | 391.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 392.30 | 392.86 | 391.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 392.30 | 392.86 | 391.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 391.95 | 392.86 | 391.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 391.00 | 392.49 | 391.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 393.15 | 391.99 | 391.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 393.45 | 392.20 | 391.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 393.50 | 392.14 | 391.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 389.90 | 391.69 | 391.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 389.90 | 391.69 | 391.81 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 392.20 | 391.71 | 391.69 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 391.00 | 391.56 | 391.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 390.50 | 391.35 | 391.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 386.85 | 386.24 | 387.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 386.85 | 386.24 | 387.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 386.85 | 386.24 | 387.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 386.85 | 386.24 | 387.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 384.90 | 385.97 | 387.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 389.40 | 385.97 | 387.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 390.40 | 386.86 | 387.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 390.30 | 386.86 | 387.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 389.85 | 387.46 | 387.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 390.10 | 387.46 | 387.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 390.50 | 388.54 | 388.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 390.90 | 389.01 | 388.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 14:15:00 | 399.90 | 404.13 | 398.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 399.90 | 404.13 | 398.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 402.00 | 403.24 | 400.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 399.95 | 403.24 | 400.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 394.50 | 401.27 | 400.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 394.50 | 401.27 | 400.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 396.10 | 400.23 | 399.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 396.50 | 400.23 | 399.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 394.15 | 399.02 | 399.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 394.15 | 399.02 | 399.23 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 405.25 | 399.84 | 399.54 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 398.85 | 399.75 | 399.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 390.15 | 397.16 | 398.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 380.30 | 379.92 | 383.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 380.30 | 379.92 | 383.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 386.55 | 381.03 | 383.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 386.55 | 381.03 | 383.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 391.30 | 383.08 | 384.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:15:00 | 392.60 | 383.08 | 384.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 391.00 | 385.80 | 385.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 404.05 | 395.16 | 393.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 411.30 | 413.96 | 409.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:45:00 | 411.80 | 413.96 | 409.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 408.00 | 412.77 | 409.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 408.50 | 412.77 | 409.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 406.30 | 411.47 | 409.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 406.30 | 411.47 | 409.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 403.00 | 407.90 | 408.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 398.00 | 402.24 | 404.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 399.55 | 399.36 | 401.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:30:00 | 397.95 | 399.36 | 401.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 402.00 | 399.68 | 401.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 402.10 | 399.68 | 401.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 399.90 | 399.73 | 401.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:30:00 | 400.40 | 399.73 | 401.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 402.00 | 400.18 | 401.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 400.75 | 400.18 | 401.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 403.20 | 400.78 | 401.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 403.20 | 400.78 | 401.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 401.65 | 400.96 | 401.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 402.20 | 400.96 | 401.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 400.35 | 399.37 | 400.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 398.35 | 399.27 | 399.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 398.15 | 398.78 | 399.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 398.05 | 398.78 | 399.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 396.20 | 398.34 | 399.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 397.30 | 396.58 | 397.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:15:00 | 400.45 | 396.58 | 397.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 400.20 | 397.31 | 398.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 400.75 | 397.31 | 398.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 394.50 | 396.75 | 397.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 390.70 | 395.44 | 397.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 378.43 | 383.35 | 388.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 378.24 | 383.35 | 388.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 378.15 | 383.35 | 388.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 376.39 | 383.35 | 388.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 392.80 | 380.28 | 383.34 | SL hit (close>ema200) qty=0.50 sl=380.28 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 395.40 | 385.66 | 385.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 400.60 | 390.36 | 387.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 398.05 | 398.49 | 392.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 14:15:00 | 414.30 | 401.16 | 395.80 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 14:45:00 | 420.25 | 404.43 | 397.78 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 09:15:00 | 416.05 | 405.75 | 398.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 11:15:00 | 414.60 | 408.77 | 401.63 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 408.45 | 411.49 | 406.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 405.35 | 411.49 | 406.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 407.60 | 410.71 | 406.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 407.35 | 410.71 | 406.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 408.70 | 410.31 | 406.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 408.45 | 410.31 | 406.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 407.25 | 409.70 | 406.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:45:00 | 406.10 | 409.70 | 406.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 407.35 | 409.23 | 406.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:15:00 | 406.65 | 409.23 | 406.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 407.65 | 408.91 | 406.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 410.00 | 408.91 | 406.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 409.75 | 409.64 | 407.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 12:45:00 | 410.25 | 410.01 | 408.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 412.50 | 415.79 | 414.46 | SL hit (close<ema400) qty=1.00 sl=414.46 alert=retest1 |

### Cycle 209 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 411.75 | 415.55 | 415.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 405.65 | 413.57 | 414.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 418.60 | 412.20 | 413.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 418.60 | 412.20 | 413.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 418.60 | 412.20 | 413.15 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 420.15 | 413.79 | 413.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 13:15:00 | 427.25 | 417.73 | 415.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 429.00 | 429.12 | 423.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:30:00 | 429.05 | 429.12 | 423.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 433.55 | 432.16 | 427.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 428.65 | 432.16 | 427.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 444.60 | 441.10 | 436.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 437.75 | 441.10 | 436.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 440.35 | 441.85 | 438.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:45:00 | 438.75 | 441.85 | 438.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 442.00 | 441.88 | 438.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 443.35 | 441.59 | 439.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 436.10 | 439.59 | 438.82 | SL hit (close<static) qty=1.00 sl=436.45 alert=retest2 |

### Cycle 211 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 436.10 | 438.00 | 438.21 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 444.00 | 439.20 | 438.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 09:15:00 | 447.10 | 442.96 | 441.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 09:15:00 | 445.35 | 446.10 | 444.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 445.35 | 446.10 | 444.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 445.35 | 446.10 | 444.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 445.20 | 446.10 | 444.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 445.90 | 446.06 | 444.19 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 435.85 | 442.12 | 442.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 15:15:00 | 434.00 | 437.67 | 440.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 438.10 | 437.67 | 439.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:00:00 | 438.10 | 437.67 | 439.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 437.40 | 437.50 | 438.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 441.45 | 437.50 | 438.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 440.50 | 438.10 | 439.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 441.70 | 438.10 | 439.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 440.35 | 438.55 | 439.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 440.35 | 438.55 | 439.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 442.35 | 439.31 | 439.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 442.35 | 439.31 | 439.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 445.00 | 440.45 | 439.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 446.85 | 441.73 | 440.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 451.85 | 452.53 | 448.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 451.85 | 452.53 | 448.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 448.50 | 451.72 | 448.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 448.70 | 451.72 | 448.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 450.45 | 451.47 | 448.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 450.75 | 451.47 | 448.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 451.25 | 451.42 | 448.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 450.35 | 451.42 | 448.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 454.00 | 453.31 | 451.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 452.45 | 453.31 | 451.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 456.15 | 455.19 | 452.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:00:00 | 468.75 | 457.77 | 454.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:30:00 | 464.55 | 463.59 | 458.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 455.85 | 459.00 | 459.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 12:15:00 | 455.85 | 459.00 | 459.35 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 465.25 | 460.30 | 459.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 469.60 | 463.31 | 461.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 464.60 | 465.46 | 463.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 464.60 | 465.46 | 463.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 464.60 | 465.46 | 463.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 464.60 | 465.46 | 463.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 477.75 | 472.89 | 469.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 471.25 | 472.89 | 469.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 469.05 | 475.65 | 473.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 469.05 | 475.65 | 473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 470.30 | 474.58 | 472.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 466.35 | 474.58 | 472.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 468.20 | 471.81 | 471.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 467.45 | 469.78 | 470.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 460.00 | 458.19 | 462.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 460.30 | 458.19 | 462.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 459.20 | 458.39 | 462.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 455.50 | 457.77 | 460.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 432.72 | 444.80 | 452.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 452.50 | 443.97 | 450.31 | SL hit (close>ema200) qty=0.50 sl=443.97 alert=retest2 |

### Cycle 218 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 450.20 | 445.57 | 445.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 452.60 | 446.98 | 446.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 451.85 | 452.28 | 449.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 09:15:00 | 466.00 | 452.28 | 449.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 448.50 | 454.21 | 451.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 448.50 | 454.21 | 451.50 | SL hit (close<ema400) qty=1.00 sl=451.50 alert=retest1 |

### Cycle 219 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 433.35 | 448.94 | 449.51 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 461.35 | 449.99 | 449.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 469.30 | 460.53 | 456.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 453.60 | 461.96 | 459.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 453.60 | 461.96 | 459.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 453.60 | 461.96 | 459.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 453.60 | 461.96 | 459.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 453.75 | 460.31 | 459.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 452.95 | 460.31 | 459.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 449.15 | 458.08 | 458.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 447.90 | 452.99 | 455.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 454.35 | 451.32 | 453.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 454.35 | 451.32 | 453.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 454.35 | 451.32 | 453.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 454.35 | 451.32 | 453.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 452.00 | 451.46 | 453.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 452.35 | 451.46 | 453.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 457.45 | 452.66 | 453.69 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 457.45 | 454.86 | 454.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 462.60 | 458.20 | 456.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 457.60 | 461.07 | 458.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 457.60 | 461.07 | 458.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 457.60 | 461.07 | 458.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 457.60 | 461.07 | 458.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 457.35 | 460.33 | 458.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 457.35 | 460.33 | 458.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 456.95 | 459.65 | 458.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 456.55 | 459.65 | 458.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 454.30 | 457.84 | 457.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 448.45 | 453.03 | 455.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 436.05 | 435.90 | 441.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 436.05 | 435.90 | 441.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 433.00 | 433.63 | 436.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 431.55 | 433.16 | 436.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:00:00 | 431.50 | 432.09 | 434.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 429.35 | 430.97 | 433.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 426.40 | 424.92 | 424.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 426.40 | 424.92 | 424.90 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 421.35 | 424.22 | 424.59 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 427.10 | 424.74 | 424.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 428.70 | 425.53 | 424.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 425.10 | 425.70 | 425.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 425.10 | 425.70 | 425.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 425.10 | 425.70 | 425.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 425.10 | 425.70 | 425.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 424.95 | 425.55 | 425.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 419.40 | 425.55 | 425.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 411.90 | 422.82 | 423.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 410.25 | 420.31 | 422.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 392.35 | 390.92 | 396.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:45:00 | 392.50 | 390.92 | 396.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 393.85 | 391.88 | 395.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 395.65 | 391.88 | 395.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 392.10 | 391.93 | 395.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:00:00 | 389.80 | 391.50 | 394.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 370.31 | 382.72 | 388.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 371.65 | 370.14 | 375.29 | SL hit (close>ema200) qty=0.50 sl=370.14 alert=retest2 |

### Cycle 228 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 367.25 | 359.15 | 358.98 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 356.45 | 361.52 | 361.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 355.15 | 359.42 | 360.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 359.95 | 359.52 | 360.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 359.95 | 359.52 | 360.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 359.95 | 359.52 | 360.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 359.45 | 359.52 | 360.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 361.00 | 359.82 | 360.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 359.55 | 360.23 | 360.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 364.65 | 356.85 | 356.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 364.65 | 356.85 | 356.10 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 351.95 | 356.88 | 357.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 345.75 | 350.66 | 353.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 356.15 | 349.73 | 351.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 356.15 | 349.73 | 351.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 356.15 | 349.73 | 351.93 | EMA400 retest candle locked (from downside) |

### Cycle 232 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 357.70 | 353.19 | 352.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 360.30 | 355.08 | 353.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 377.75 | 382.63 | 379.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 377.75 | 382.63 | 379.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 377.75 | 382.63 | 379.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 386.85 | 380.81 | 379.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 13:15:00 | 425.54 | 419.46 | 416.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 413.40 | 418.76 | 419.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 407.60 | 414.66 | 416.94 | Break + close below crossover candle low |

### Cycle 234 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 439.45 | 418.71 | 418.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 456.85 | 426.34 | 421.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 15:15:00 | 428.40 | 435.37 | 429.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 15:15:00 | 428.40 | 435.37 | 429.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 428.40 | 435.37 | 429.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 426.50 | 433.60 | 428.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 426.65 | 432.21 | 428.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:15:00 | 423.60 | 432.21 | 428.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 421.30 | 430.03 | 428.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 421.30 | 430.03 | 428.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 235 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 417.55 | 425.77 | 426.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 404.45 | 419.75 | 423.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 15:15:00 | 395.00 | 393.11 | 398.55 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:15:00 | 391.75 | 393.11 | 398.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 395.20 | 393.42 | 397.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 395.65 | 393.42 | 397.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 395.60 | 394.72 | 397.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 395.60 | 394.72 | 397.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 392.55 | 394.37 | 396.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 397.60 | 394.88 | 396.17 | SL hit (close>ema400) qty=1.00 sl=396.17 alert=retest1 |

### Cycle 236 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 400.05 | 396.97 | 396.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 411.10 | 400.60 | 398.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 409.15 | 409.17 | 405.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 409.80 | 409.17 | 405.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 406.35 | 408.07 | 405.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 406.40 | 408.07 | 405.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 406.85 | 407.83 | 405.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 406.30 | 407.83 | 405.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-14 12:45:00 | 230.05 | 2023-06-15 10:15:00 | 225.45 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2023-06-16 13:30:00 | 223.50 | 2023-06-21 09:15:00 | 224.75 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-06-16 14:15:00 | 224.00 | 2023-06-21 09:15:00 | 224.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-06-19 09:15:00 | 222.20 | 2023-06-21 11:15:00 | 225.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-06-20 10:15:00 | 223.80 | 2023-06-21 11:15:00 | 225.70 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-06-20 11:15:00 | 222.20 | 2023-06-21 11:15:00 | 225.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-06-20 14:45:00 | 222.40 | 2023-06-21 11:15:00 | 225.70 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-06-27 11:30:00 | 216.00 | 2023-07-07 09:15:00 | 214.45 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2023-08-16 12:30:00 | 247.50 | 2023-08-17 10:15:00 | 243.85 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-09-11 15:15:00 | 249.80 | 2023-09-12 09:15:00 | 237.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:30:00 | 247.25 | 2023-09-12 14:15:00 | 234.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 15:15:00 | 249.80 | 2023-09-13 10:15:00 | 241.35 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2023-09-12 09:30:00 | 247.25 | 2023-09-13 10:15:00 | 241.35 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2023-09-14 09:30:00 | 249.40 | 2023-09-14 10:15:00 | 253.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-09-26 12:30:00 | 234.25 | 2023-09-29 14:15:00 | 232.55 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2023-09-29 13:15:00 | 233.90 | 2023-09-29 14:15:00 | 232.55 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2023-10-12 09:15:00 | 229.75 | 2023-10-12 15:15:00 | 228.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-10-12 11:30:00 | 229.90 | 2023-10-12 15:15:00 | 228.80 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-10-12 13:15:00 | 229.55 | 2023-10-12 15:15:00 | 228.80 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2023-10-18 15:00:00 | 220.80 | 2023-10-19 14:15:00 | 225.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2023-10-19 09:30:00 | 220.25 | 2023-10-19 14:15:00 | 225.50 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2023-10-20 13:30:00 | 225.65 | 2023-10-23 09:15:00 | 217.70 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2023-10-20 14:15:00 | 225.80 | 2023-10-23 09:15:00 | 217.70 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2023-10-20 15:00:00 | 226.30 | 2023-10-23 09:15:00 | 217.70 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2023-10-27 10:15:00 | 207.90 | 2023-10-31 13:15:00 | 210.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-10-27 12:00:00 | 207.70 | 2023-10-31 13:15:00 | 210.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-10-27 12:30:00 | 207.70 | 2023-10-31 13:15:00 | 210.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-10-30 09:30:00 | 207.40 | 2023-10-31 13:15:00 | 210.40 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-11-09 13:30:00 | 214.80 | 2023-11-09 14:15:00 | 212.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-11-20 12:45:00 | 219.30 | 2023-11-22 12:15:00 | 215.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-11-21 09:15:00 | 219.15 | 2023-11-22 12:15:00 | 215.65 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-11-21 10:15:00 | 219.65 | 2023-11-22 12:15:00 | 215.65 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-11-21 12:15:00 | 218.90 | 2023-11-22 12:15:00 | 215.65 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-12-05 15:00:00 | 256.20 | 2023-12-15 14:15:00 | 265.95 | STOP_HIT | 1.00 | 3.81% |
| BUY | retest2 | 2023-12-07 09:30:00 | 250.60 | 2023-12-15 14:15:00 | 265.95 | STOP_HIT | 1.00 | 6.13% |
| BUY | retest2 | 2023-12-07 11:45:00 | 250.75 | 2023-12-15 14:15:00 | 265.95 | STOP_HIT | 1.00 | 6.06% |
| BUY | retest2 | 2023-12-29 11:45:00 | 261.80 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-01-01 09:15:00 | 261.75 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-01-02 10:00:00 | 261.15 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-01-02 14:15:00 | 261.10 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-01-03 09:15:00 | 265.60 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-01-03 09:45:00 | 263.55 | 2024-01-08 09:15:00 | 260.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-01-09 14:15:00 | 258.80 | 2024-01-12 12:15:00 | 261.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-01-10 14:15:00 | 259.95 | 2024-01-12 12:15:00 | 261.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-01-11 09:30:00 | 259.95 | 2024-01-12 12:15:00 | 261.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-01-16 12:45:00 | 258.05 | 2024-01-18 09:15:00 | 245.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 15:15:00 | 256.50 | 2024-01-18 09:15:00 | 243.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 12:45:00 | 258.05 | 2024-01-19 09:15:00 | 253.75 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2024-01-16 15:15:00 | 256.50 | 2024-01-19 09:15:00 | 253.75 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-02-08 11:00:00 | 238.05 | 2024-02-09 11:15:00 | 241.95 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-02-08 14:00:00 | 238.40 | 2024-02-09 11:15:00 | 241.95 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-02-19 14:30:00 | 242.30 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-02-19 15:15:00 | 241.95 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-02-20 10:45:00 | 241.75 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-02-20 12:00:00 | 241.75 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-02-20 15:15:00 | 244.15 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-02-22 09:30:00 | 242.20 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-02-22 10:00:00 | 244.80 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-02-26 10:00:00 | 242.10 | 2024-02-26 10:15:00 | 241.75 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-03-05 09:15:00 | 233.55 | 2024-03-05 09:15:00 | 231.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-03-11 11:00:00 | 223.45 | 2024-03-12 14:15:00 | 212.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 11:45:00 | 222.80 | 2024-03-12 15:15:00 | 211.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 11:00:00 | 223.45 | 2024-03-13 12:15:00 | 201.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-11 11:45:00 | 222.80 | 2024-03-13 12:15:00 | 200.52 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-03-28 09:15:00 | 213.45 | 2024-04-02 14:15:00 | 224.12 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-28 15:15:00 | 213.40 | 2024-04-02 14:15:00 | 224.07 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-28 09:15:00 | 213.45 | 2024-04-03 13:15:00 | 223.40 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest1 | 2024-03-28 15:15:00 | 213.40 | 2024-04-03 13:15:00 | 223.40 | STOP_HIT | 0.50 | 4.69% |
| BUY | retest2 | 2024-04-01 09:15:00 | 216.15 | 2024-04-10 11:15:00 | 225.85 | STOP_HIT | 1.00 | 4.49% |
| SELL | retest2 | 2024-04-22 13:15:00 | 220.85 | 2024-04-23 13:15:00 | 225.55 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-04-22 14:00:00 | 220.80 | 2024-04-23 13:15:00 | 225.55 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-04-23 10:30:00 | 220.15 | 2024-04-23 13:15:00 | 225.55 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-04-23 11:15:00 | 221.00 | 2024-04-23 13:15:00 | 225.55 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-04-26 09:15:00 | 228.15 | 2024-04-30 14:15:00 | 224.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-04-26 11:00:00 | 228.40 | 2024-04-30 14:15:00 | 224.90 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-04-29 09:15:00 | 228.95 | 2024-04-30 14:15:00 | 224.90 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-04-29 12:00:00 | 228.25 | 2024-04-30 14:15:00 | 224.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-04-30 09:15:00 | 227.30 | 2024-04-30 14:15:00 | 224.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-05-03 11:15:00 | 224.50 | 2024-05-07 10:15:00 | 213.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:15:00 | 224.50 | 2024-05-08 09:15:00 | 210.90 | STOP_HIT | 0.50 | 6.06% |
| BUY | retest2 | 2024-05-16 09:15:00 | 209.25 | 2024-05-16 13:15:00 | 207.35 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-05-16 10:30:00 | 209.50 | 2024-05-16 13:15:00 | 207.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-05-16 12:30:00 | 209.35 | 2024-05-16 13:15:00 | 207.35 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-05-17 09:30:00 | 210.35 | 2024-05-22 10:15:00 | 209.75 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-05-24 10:00:00 | 208.40 | 2024-05-24 13:15:00 | 210.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-05-24 15:00:00 | 208.80 | 2024-05-27 09:15:00 | 212.75 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-06-14 09:15:00 | 221.40 | 2024-06-19 13:15:00 | 219.82 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-06-18 10:45:00 | 220.50 | 2024-06-19 13:15:00 | 219.82 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-06-19 09:30:00 | 220.60 | 2024-06-19 13:15:00 | 219.82 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-06-19 12:00:00 | 220.87 | 2024-06-19 13:15:00 | 219.82 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-06-24 12:30:00 | 236.50 | 2024-06-26 09:15:00 | 260.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 09:15:00 | 242.77 | 2024-06-26 11:15:00 | 267.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-10 10:15:00 | 276.65 | 2024-07-10 11:15:00 | 284.75 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-07-15 11:15:00 | 306.35 | 2024-07-18 09:15:00 | 336.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-13 09:15:00 | 366.40 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-08-13 09:45:00 | 365.80 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-13 12:30:00 | 365.65 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-08-19 09:45:00 | 366.00 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-08-20 10:15:00 | 363.95 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-08-20 11:45:00 | 364.00 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-08-20 12:45:00 | 364.00 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-08-21 13:15:00 | 363.85 | 2024-08-22 10:15:00 | 366.55 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-08-30 13:15:00 | 365.05 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-09-02 13:30:00 | 365.15 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-02 14:00:00 | 365.10 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-09-03 10:00:00 | 365.30 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-09-04 10:15:00 | 364.35 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-09-04 12:00:00 | 364.40 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-09-04 13:30:00 | 364.45 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-09-04 14:45:00 | 364.55 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-09-06 09:15:00 | 363.55 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-06 11:00:00 | 363.45 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-06 12:30:00 | 363.95 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-09 11:45:00 | 363.05 | 2024-09-09 15:15:00 | 366.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-09-23 14:30:00 | 365.70 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-09-24 09:15:00 | 365.25 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-09-24 10:00:00 | 365.25 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-09-26 09:15:00 | 365.20 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-09-27 14:45:00 | 359.60 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-30 12:15:00 | 360.20 | 2024-09-30 14:15:00 | 365.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-03 14:45:00 | 366.20 | 2024-10-04 11:15:00 | 363.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-10-04 09:45:00 | 364.95 | 2024-10-04 11:15:00 | 363.05 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-08 09:15:00 | 360.60 | 2024-10-10 09:15:00 | 363.55 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-10-09 10:00:00 | 361.30 | 2024-10-10 09:15:00 | 363.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-10-09 11:00:00 | 361.30 | 2024-10-10 09:15:00 | 363.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-10-18 15:15:00 | 361.60 | 2024-10-28 13:15:00 | 360.40 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-11-11 09:15:00 | 357.00 | 2024-11-18 15:15:00 | 357.10 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-11-11 13:30:00 | 357.95 | 2024-11-18 15:15:00 | 357.10 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-11-12 12:00:00 | 357.55 | 2024-11-18 15:15:00 | 357.10 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-11-12 12:45:00 | 358.00 | 2024-11-18 15:15:00 | 357.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-11-27 12:30:00 | 361.50 | 2024-12-05 15:15:00 | 366.30 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2024-11-28 12:30:00 | 361.20 | 2024-12-05 15:15:00 | 366.30 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2024-12-12 09:15:00 | 344.05 | 2024-12-19 09:15:00 | 341.20 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-12-16 09:15:00 | 342.10 | 2024-12-19 09:15:00 | 341.20 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-01-01 09:15:00 | 379.35 | 2025-01-06 13:15:00 | 376.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-01-01 15:00:00 | 377.50 | 2025-01-06 13:15:00 | 376.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-01-02 15:15:00 | 377.50 | 2025-01-06 13:15:00 | 376.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-01-03 11:30:00 | 377.60 | 2025-01-06 13:15:00 | 376.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-01-17 14:15:00 | 379.60 | 2025-01-20 09:15:00 | 377.55 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-01-29 13:30:00 | 278.40 | 2025-01-31 09:15:00 | 264.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-29 13:30:00 | 278.40 | 2025-02-01 09:15:00 | 275.80 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2025-01-30 09:15:00 | 271.75 | 2025-02-04 09:15:00 | 258.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 09:15:00 | 271.75 | 2025-02-04 09:15:00 | 270.50 | STOP_HIT | 0.50 | 0.46% |
| BUY | retest2 | 2025-02-07 11:15:00 | 285.60 | 2025-02-10 09:15:00 | 274.35 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-02-19 13:15:00 | 268.80 | 2025-02-24 09:15:00 | 255.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 13:45:00 | 269.10 | 2025-02-24 09:15:00 | 255.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 14:15:00 | 268.60 | 2025-02-24 09:15:00 | 255.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 15:00:00 | 266.60 | 2025-02-24 09:15:00 | 253.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 13:15:00 | 268.80 | 2025-02-25 15:15:00 | 255.30 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-02-19 13:45:00 | 269.10 | 2025-02-25 15:15:00 | 255.30 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2025-02-19 14:15:00 | 268.60 | 2025-02-25 15:15:00 | 255.30 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-02-19 15:00:00 | 266.60 | 2025-02-25 15:15:00 | 255.30 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-02-21 10:15:00 | 259.00 | 2025-02-27 11:15:00 | 246.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 13:30:00 | 259.20 | 2025-02-27 11:15:00 | 246.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 15:15:00 | 258.80 | 2025-02-27 11:15:00 | 245.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:15:00 | 259.00 | 2025-02-28 14:15:00 | 255.20 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2025-02-21 13:30:00 | 259.20 | 2025-02-28 14:15:00 | 255.20 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2025-02-21 15:15:00 | 258.80 | 2025-02-28 14:15:00 | 255.20 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2025-03-13 09:15:00 | 276.20 | 2025-03-18 14:15:00 | 275.00 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest1 | 2025-03-24 09:15:00 | 288.10 | 2025-03-25 09:15:00 | 278.40 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-04-02 12:00:00 | 279.80 | 2025-04-04 12:15:00 | 276.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-04-02 13:15:00 | 279.95 | 2025-04-04 12:15:00 | 276.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-04-02 15:15:00 | 282.80 | 2025-04-04 12:15:00 | 276.60 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-04-07 09:15:00 | 265.30 | 2025-04-11 09:15:00 | 280.25 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-04-08 09:30:00 | 274.50 | 2025-04-11 09:15:00 | 280.25 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-04-17 09:15:00 | 302.70 | 2025-04-23 10:15:00 | 285.00 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2025-05-05 09:15:00 | 306.15 | 2025-05-09 10:15:00 | 305.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-05-27 11:30:00 | 330.40 | 2025-05-30 13:15:00 | 330.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-06-09 14:45:00 | 342.50 | 2025-06-11 14:15:00 | 340.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-10 10:00:00 | 344.00 | 2025-06-11 14:15:00 | 340.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-10 12:00:00 | 346.45 | 2025-06-11 14:15:00 | 340.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-06-18 12:30:00 | 325.55 | 2025-06-20 14:15:00 | 309.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 13:45:00 | 325.40 | 2025-06-20 14:15:00 | 309.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:30:00 | 325.55 | 2025-06-23 12:15:00 | 311.70 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-06-18 13:45:00 | 325.40 | 2025-06-23 12:15:00 | 311.70 | STOP_HIT | 0.50 | 4.21% |
| BUY | retest2 | 2025-06-30 13:15:00 | 339.00 | 2025-07-04 13:15:00 | 344.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-07-25 11:45:00 | 358.65 | 2025-07-25 15:15:00 | 357.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-07-25 12:30:00 | 359.90 | 2025-07-25 15:15:00 | 357.75 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-09-01 10:00:00 | 387.45 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2025-09-01 15:00:00 | 384.25 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | 2.52% |
| BUY | retest2 | 2025-09-02 09:15:00 | 384.00 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-09-02 11:45:00 | 384.00 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-09-05 13:15:00 | 396.65 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-08 09:15:00 | 405.45 | 2025-09-09 12:15:00 | 393.95 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-09-25 13:00:00 | 386.60 | 2025-10-01 13:15:00 | 386.85 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-09-25 13:30:00 | 386.90 | 2025-10-01 13:15:00 | 386.85 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-09-26 09:15:00 | 386.45 | 2025-10-01 13:15:00 | 386.85 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-26 15:00:00 | 384.65 | 2025-10-01 13:15:00 | 386.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-29 14:45:00 | 383.80 | 2025-10-01 13:15:00 | 386.85 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-13 09:15:00 | 393.20 | 2025-10-14 10:15:00 | 389.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-14 09:15:00 | 392.65 | 2025-10-14 10:15:00 | 389.30 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-20 09:15:00 | 393.15 | 2025-10-21 14:15:00 | 389.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-20 10:30:00 | 393.45 | 2025-10-21 14:15:00 | 389.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-20 15:15:00 | 393.50 | 2025-10-21 14:15:00 | 389.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-03 11:15:00 | 396.50 | 2025-11-03 11:15:00 | 394.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-27 15:00:00 | 398.35 | 2025-12-03 09:15:00 | 378.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:30:00 | 398.15 | 2025-12-03 09:15:00 | 378.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 11:15:00 | 398.05 | 2025-12-03 09:15:00 | 378.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 12:30:00 | 396.20 | 2025-12-03 09:15:00 | 376.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 15:00:00 | 398.35 | 2025-12-04 09:15:00 | 392.80 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-28 10:30:00 | 398.15 | 2025-12-04 09:15:00 | 392.80 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2025-11-28 11:15:00 | 398.05 | 2025-12-04 09:15:00 | 392.80 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-11-28 12:30:00 | 396.20 | 2025-12-04 09:15:00 | 392.80 | STOP_HIT | 0.50 | 0.86% |
| SELL | retest2 | 2025-12-01 14:00:00 | 390.70 | 2025-12-04 11:15:00 | 395.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-12-04 10:30:00 | 393.50 | 2025-12-04 11:15:00 | 395.40 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-05 14:15:00 | 414.30 | 2025-12-12 13:15:00 | 412.50 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-12-05 14:45:00 | 420.25 | 2025-12-12 13:15:00 | 412.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2025-12-08 09:15:00 | 416.05 | 2025-12-12 13:15:00 | 412.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2025-12-08 11:15:00 | 414.60 | 2025-12-12 13:15:00 | 412.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-09 15:15:00 | 410.00 | 2025-12-16 10:15:00 | 411.75 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-12-10 11:15:00 | 409.75 | 2025-12-16 10:15:00 | 411.75 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-12-10 12:45:00 | 410.25 | 2025-12-16 10:15:00 | 411.75 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-12-24 09:15:00 | 443.35 | 2025-12-24 12:15:00 | 436.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-08 14:00:00 | 468.75 | 2026-01-12 12:15:00 | 455.85 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-01-09 09:30:00 | 464.55 | 2026-01-12 12:15:00 | 455.85 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-01-23 09:15:00 | 455.50 | 2026-01-23 14:15:00 | 432.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 455.50 | 2026-01-27 09:15:00 | 452.50 | STOP_HIT | 0.50 | 0.66% |
| SELL | retest2 | 2026-01-27 14:00:00 | 452.85 | 2026-01-30 10:15:00 | 450.20 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest1 | 2026-02-01 09:15:00 | 466.00 | 2026-02-01 12:15:00 | 448.50 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2026-02-18 10:45:00 | 431.55 | 2026-02-25 09:15:00 | 426.40 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-02-18 14:00:00 | 431.50 | 2026-02-25 09:15:00 | 426.40 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2026-02-19 10:00:00 | 429.35 | 2026-02-25 09:15:00 | 426.40 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-03-06 10:00:00 | 389.80 | 2026-03-09 09:15:00 | 370.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:00:00 | 389.80 | 2026-03-10 13:15:00 | 371.65 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2026-03-20 15:00:00 | 359.55 | 2026-03-25 09:15:00 | 364.65 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-15 09:15:00 | 386.85 | 2026-04-22 13:15:00 | 425.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-05-05 09:15:00 | 391.75 | 2026-05-06 12:15:00 | 397.60 | STOP_HIT | 1.00 | -1.49% |
