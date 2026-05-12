# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 362.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 153 |
| ALERT1 | 99 |
| ALERT2 | 98 |
| ALERT2_SKIP | 45 |
| ALERT3 | 232 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 110 |
| PARTIAL | 21 |
| TARGET_HIT | 8 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 66 / 67
- **Target hits / Stop hits / Partials:** 8 / 104 / 21
- **Avg / median % per leg:** 1.47% / 0.00%
- **Sum % (uncompounded):** 195.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 23 | 39.0% | 8 | 50 | 1 | 1.14% | 67.1% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.45% | 13.8% |
| BUY @ 3rd Alert (retest2) | 55 | 20 | 36.4% | 7 | 48 | 0 | 0.97% | 53.3% |
| SELL (all) | 74 | 43 | 58.1% | 0 | 54 | 20 | 1.74% | 128.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 74 | 43 | 58.1% | 0 | 54 | 20 | 1.74% | 128.6% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.45% | 13.8% |
| retest2 (combined) | 129 | 63 | 48.8% | 7 | 102 | 20 | 1.41% | 181.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 222.55 | 217.29 | 216.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 227.35 | 222.28 | 221.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 15:15:00 | 224.80 | 225.23 | 223.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 09:15:00 | 223.55 | 225.23 | 223.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 224.00 | 224.99 | 223.53 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 222.50 | 223.92 | 223.99 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 225.85 | 224.24 | 224.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 15:15:00 | 227.00 | 225.49 | 224.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 228.10 | 228.21 | 226.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 228.10 | 228.21 | 226.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 228.10 | 228.21 | 226.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 228.10 | 228.21 | 226.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 226.90 | 228.91 | 227.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 226.90 | 228.91 | 227.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 225.50 | 228.23 | 227.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 228.10 | 228.23 | 227.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 230.90 | 228.61 | 227.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:45:00 | 228.25 | 228.62 | 228.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:30:00 | 228.30 | 228.47 | 228.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 227.25 | 228.22 | 228.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 227.25 | 228.22 | 228.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 227.05 | 227.99 | 227.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 226.95 | 227.99 | 227.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 227.00 | 227.79 | 227.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 227.00 | 227.79 | 227.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 222.70 | 226.22 | 227.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 223.40 | 223.18 | 224.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 10:45:00 | 222.75 | 223.18 | 224.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 225.35 | 223.51 | 224.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 225.35 | 223.51 | 224.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 223.40 | 223.49 | 224.34 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 229.75 | 225.64 | 225.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 231.00 | 227.38 | 226.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 223.65 | 227.47 | 226.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 223.65 | 227.47 | 226.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 223.65 | 227.47 | 226.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 219.60 | 227.47 | 226.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 207.05 | 223.39 | 224.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 200.15 | 218.74 | 222.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 214.60 | 212.88 | 217.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 214.60 | 212.88 | 217.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 214.60 | 212.82 | 216.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 214.60 | 212.82 | 216.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 216.15 | 213.48 | 216.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:30:00 | 217.30 | 213.48 | 216.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 215.00 | 213.79 | 216.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 220.50 | 213.79 | 216.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 220.75 | 215.18 | 216.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 220.75 | 215.18 | 216.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 221.35 | 217.69 | 217.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 224.45 | 220.31 | 218.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 232.24 | 232.78 | 229.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 232.30 | 232.78 | 229.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 237.00 | 239.50 | 238.33 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 235.59 | 237.33 | 237.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 233.10 | 236.18 | 236.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 12:15:00 | 238.40 | 236.37 | 236.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 12:15:00 | 238.40 | 236.37 | 236.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 238.40 | 236.37 | 236.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:30:00 | 239.74 | 236.37 | 236.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 240.10 | 237.11 | 237.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 245.86 | 239.13 | 238.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 241.75 | 243.16 | 241.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 12:15:00 | 241.75 | 243.16 | 241.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 241.75 | 243.16 | 241.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 241.75 | 243.16 | 241.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 243.14 | 243.15 | 241.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:15:00 | 241.10 | 243.15 | 241.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 241.51 | 242.83 | 241.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 241.30 | 242.83 | 241.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 239.90 | 242.24 | 241.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 238.79 | 242.24 | 241.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 237.89 | 241.37 | 241.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 236.52 | 241.37 | 241.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 239.18 | 240.93 | 241.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 237.00 | 238.85 | 239.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 15:15:00 | 238.00 | 238.00 | 238.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 09:15:00 | 236.73 | 238.00 | 238.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 236.50 | 237.70 | 238.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 15:15:00 | 235.15 | 238.03 | 238.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 234.85 | 236.78 | 237.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 10:15:00 | 242.17 | 237.52 | 237.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 242.17 | 237.52 | 237.40 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 235.34 | 238.05 | 238.36 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 237.74 | 237.62 | 237.62 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 15:15:00 | 237.50 | 237.60 | 237.61 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 238.46 | 237.77 | 237.68 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 237.00 | 237.62 | 237.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 234.12 | 236.48 | 237.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 229.80 | 228.87 | 230.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 229.80 | 228.87 | 230.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 228.66 | 228.85 | 230.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 227.95 | 228.85 | 230.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 216.55 | 221.41 | 223.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 13:15:00 | 217.60 | 217.56 | 219.59 | SL hit (close>ema200) qty=0.50 sl=217.56 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 218.67 | 216.44 | 216.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 220.95 | 217.78 | 217.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 222.82 | 222.90 | 221.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 222.82 | 222.90 | 221.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 228.36 | 226.44 | 224.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 228.10 | 226.44 | 224.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 224.55 | 226.08 | 225.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 224.55 | 226.08 | 225.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 220.35 | 224.93 | 224.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 220.70 | 224.93 | 224.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 219.95 | 223.94 | 224.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 213.59 | 221.87 | 223.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 205.44 | 205.11 | 208.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 206.25 | 205.11 | 208.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 209.47 | 206.19 | 207.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 209.47 | 206.19 | 207.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 211.60 | 207.27 | 208.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 211.60 | 207.27 | 208.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 211.55 | 208.82 | 208.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 212.50 | 210.08 | 209.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 210.95 | 211.10 | 210.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 210.95 | 211.10 | 210.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 210.20 | 210.92 | 210.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:30:00 | 211.35 | 210.91 | 210.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 211.10 | 210.95 | 210.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 11:45:00 | 212.50 | 211.35 | 210.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:30:00 | 211.50 | 211.73 | 211.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 211.75 | 211.73 | 211.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:00:00 | 213.98 | 212.18 | 211.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 214.22 | 212.44 | 211.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 210.36 | 212.15 | 211.83 | SL hit (close<static) qty=1.00 sl=211.05 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 209.26 | 211.57 | 211.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 207.91 | 210.84 | 211.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 09:15:00 | 210.20 | 210.13 | 210.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 10:00:00 | 210.20 | 210.13 | 210.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 209.78 | 207.29 | 208.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 209.78 | 207.29 | 208.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 209.90 | 207.81 | 208.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:15:00 | 210.82 | 207.81 | 208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 213.97 | 209.85 | 209.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 215.29 | 210.94 | 210.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 11:15:00 | 213.39 | 213.48 | 211.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 11:45:00 | 213.20 | 213.48 | 211.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 213.00 | 213.38 | 212.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 213.00 | 213.38 | 212.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 222.00 | 222.80 | 221.36 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 218.71 | 220.91 | 220.93 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 221.26 | 220.98 | 220.96 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 220.32 | 220.85 | 220.90 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 221.50 | 220.93 | 220.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 221.91 | 221.22 | 221.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 11:15:00 | 222.50 | 222.70 | 222.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 12:00:00 | 222.50 | 222.70 | 222.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 222.40 | 222.64 | 222.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:30:00 | 221.86 | 222.64 | 222.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 223.17 | 222.75 | 222.27 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 219.26 | 221.67 | 221.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 219.18 | 220.85 | 221.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 221.58 | 221.00 | 221.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 221.58 | 221.00 | 221.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 221.58 | 221.00 | 221.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 221.58 | 221.00 | 221.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 221.22 | 221.04 | 221.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 222.55 | 221.04 | 221.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 222.25 | 221.28 | 221.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 221.63 | 221.31 | 221.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 11:15:00 | 223.24 | 221.41 | 221.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 223.24 | 221.41 | 221.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 12:15:00 | 224.01 | 221.93 | 221.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 12:15:00 | 224.80 | 224.97 | 223.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 13:00:00 | 224.80 | 224.97 | 223.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 223.85 | 224.74 | 223.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 223.85 | 224.74 | 223.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 225.26 | 224.85 | 223.88 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 221.79 | 223.19 | 223.38 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 15:15:00 | 223.42 | 223.16 | 223.13 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 220.30 | 222.59 | 222.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 218.25 | 221.08 | 222.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 216.79 | 216.78 | 218.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 216.79 | 216.78 | 218.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 216.95 | 216.84 | 218.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 216.48 | 216.84 | 218.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 218.88 | 217.51 | 218.32 | SL hit (close>static) qty=1.00 sl=218.50 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 219.76 | 217.45 | 217.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 220.77 | 218.11 | 217.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 224.69 | 225.68 | 223.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:45:00 | 225.14 | 225.68 | 223.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 223.57 | 225.26 | 223.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 223.57 | 225.26 | 223.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 223.26 | 224.86 | 223.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:15:00 | 224.20 | 224.86 | 223.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 11:45:00 | 224.21 | 227.02 | 226.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 234.30 | 237.17 | 237.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 234.30 | 237.17 | 237.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 232.09 | 235.20 | 236.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 225.00 | 224.70 | 226.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 10:15:00 | 225.50 | 224.86 | 226.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 225.50 | 224.86 | 226.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 226.08 | 224.86 | 226.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 226.10 | 225.18 | 226.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:45:00 | 226.56 | 225.18 | 226.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 225.38 | 225.22 | 226.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:45:00 | 227.19 | 225.22 | 226.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 225.41 | 225.26 | 226.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:30:00 | 224.97 | 225.26 | 226.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 226.30 | 225.51 | 226.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:30:00 | 225.49 | 225.53 | 226.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 225.64 | 225.53 | 226.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 224.68 | 225.52 | 226.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 225.01 | 223.69 | 223.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 225.01 | 223.69 | 223.65 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 221.80 | 224.60 | 224.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 220.85 | 223.39 | 224.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 220.24 | 219.80 | 221.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 220.24 | 219.80 | 221.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 222.05 | 220.47 | 221.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 222.40 | 220.47 | 221.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 222.21 | 220.82 | 221.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 220.27 | 220.82 | 221.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 219.18 | 220.49 | 221.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 217.34 | 219.55 | 220.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 216.43 | 218.51 | 220.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 206.47 | 212.31 | 215.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 211.63 | 211.59 | 214.45 | SL hit (close>ema200) qty=0.50 sl=211.59 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 215.81 | 207.22 | 206.10 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 203.20 | 205.84 | 205.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 201.55 | 204.98 | 205.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 207.24 | 204.93 | 205.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 207.24 | 204.93 | 205.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 207.24 | 204.93 | 205.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 207.24 | 204.93 | 205.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 206.45 | 205.23 | 205.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 201.65 | 205.23 | 205.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 205.56 | 202.55 | 202.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 205.56 | 202.55 | 202.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 206.39 | 203.32 | 202.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 204.60 | 204.67 | 203.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 204.60 | 204.67 | 203.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 204.66 | 204.66 | 203.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 203.87 | 204.66 | 203.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 203.60 | 204.45 | 203.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 203.60 | 204.45 | 203.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 203.78 | 204.32 | 203.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 202.95 | 204.32 | 203.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 203.21 | 204.10 | 203.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 202.91 | 204.10 | 203.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 203.23 | 203.92 | 203.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 202.69 | 203.92 | 203.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 203.16 | 203.77 | 203.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 203.68 | 203.77 | 203.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 203.12 | 203.71 | 203.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 203.12 | 203.71 | 203.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 202.30 | 203.43 | 203.45 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 12:15:00 | 203.68 | 203.48 | 203.47 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 202.22 | 203.23 | 203.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 201.28 | 202.84 | 203.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 202.71 | 202.13 | 202.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 202.71 | 202.13 | 202.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 202.71 | 202.13 | 202.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 202.71 | 202.13 | 202.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 201.40 | 201.99 | 202.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 200.40 | 201.99 | 202.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 200.77 | 201.85 | 202.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 190.38 | 195.81 | 198.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 190.73 | 195.81 | 198.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 190.11 | 189.98 | 192.79 | SL hit (close>ema200) qty=0.50 sl=189.98 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 186.80 | 185.88 | 185.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 191.18 | 186.94 | 186.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 193.69 | 193.76 | 192.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 193.69 | 193.76 | 192.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 193.19 | 193.82 | 193.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 193.19 | 193.82 | 193.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 193.85 | 193.83 | 193.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 194.23 | 193.83 | 193.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 194.00 | 193.89 | 193.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:15:00 | 194.30 | 193.94 | 193.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 194.17 | 193.88 | 193.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 194.68 | 194.04 | 193.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 195.80 | 194.11 | 193.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 197.16 | 198.23 | 198.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 197.16 | 198.23 | 198.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 196.75 | 197.93 | 198.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 200.18 | 198.20 | 198.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 200.18 | 198.20 | 198.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 200.18 | 198.20 | 198.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 200.18 | 198.20 | 198.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 201.55 | 198.87 | 198.56 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 198.34 | 199.53 | 199.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 193.07 | 197.48 | 198.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 196.80 | 196.22 | 197.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 196.80 | 196.22 | 197.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 196.81 | 196.34 | 197.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 197.68 | 196.34 | 197.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 197.91 | 196.65 | 197.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 198.37 | 196.65 | 197.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 196.06 | 196.53 | 197.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:30:00 | 195.90 | 196.38 | 197.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 195.68 | 196.07 | 196.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 186.10 | 190.91 | 192.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 190.72 | 190.25 | 191.60 | SL hit (close>ema200) qty=0.50 sl=190.25 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 184.18 | 180.69 | 180.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 184.74 | 181.50 | 180.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 181.65 | 183.81 | 182.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 181.65 | 183.81 | 182.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 181.65 | 183.81 | 182.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 181.45 | 183.81 | 182.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 177.32 | 182.51 | 182.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 177.32 | 182.51 | 182.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 178.30 | 181.67 | 182.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 176.13 | 179.33 | 180.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 178.55 | 178.39 | 179.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 178.31 | 178.39 | 179.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 179.20 | 178.55 | 179.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 179.41 | 178.55 | 179.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 179.25 | 178.69 | 179.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:15:00 | 179.19 | 178.69 | 179.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 179.19 | 178.79 | 179.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 176.50 | 178.79 | 179.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 178.61 | 178.91 | 179.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 178.77 | 178.01 | 178.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 169.68 | 174.38 | 176.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 169.83 | 174.38 | 176.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 167.67 | 171.00 | 173.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 172.48 | 170.76 | 173.15 | SL hit (close>ema200) qty=0.50 sl=170.76 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 174.05 | 172.79 | 172.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 14:15:00 | 174.26 | 173.08 | 172.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 12:15:00 | 173.50 | 173.76 | 173.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 13:00:00 | 173.50 | 173.76 | 173.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 172.88 | 173.59 | 173.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:45:00 | 172.73 | 173.59 | 173.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 173.69 | 173.61 | 173.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 177.43 | 173.66 | 173.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 174.63 | 176.70 | 176.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 174.63 | 176.70 | 176.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 171.68 | 175.30 | 176.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 173.21 | 172.55 | 174.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 173.21 | 172.55 | 174.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 175.27 | 173.25 | 174.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 175.27 | 173.25 | 174.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 174.29 | 173.46 | 174.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 173.38 | 173.46 | 174.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 174.08 | 173.97 | 174.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 173.90 | 174.02 | 174.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 173.20 | 173.48 | 174.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 174.22 | 173.46 | 173.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 174.22 | 173.46 | 173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 173.60 | 173.49 | 173.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 172.08 | 173.04 | 173.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 164.71 | 169.70 | 171.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 165.38 | 169.70 | 171.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 165.20 | 169.70 | 171.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 164.54 | 169.70 | 171.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 163.48 | 166.27 | 168.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 167.08 | 166.11 | 168.36 | SL hit (close>ema200) qty=0.50 sl=166.11 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 175.85 | 169.73 | 169.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 176.42 | 172.02 | 170.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 175.67 | 175.83 | 173.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 175.55 | 175.83 | 173.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 178.09 | 178.54 | 176.96 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 172.00 | 176.09 | 176.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 169.12 | 173.53 | 174.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 170.91 | 169.16 | 171.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 170.91 | 169.16 | 171.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 170.91 | 169.16 | 171.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:30:00 | 167.76 | 169.36 | 170.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:45:00 | 168.10 | 168.91 | 169.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 159.37 | 160.84 | 164.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 159.69 | 160.84 | 164.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 15:15:00 | 159.75 | 159.39 | 161.85 | SL hit (close>ema200) qty=0.50 sl=159.39 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 155.65 | 154.56 | 154.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 157.95 | 155.75 | 155.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 156.60 | 157.12 | 156.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 156.60 | 157.12 | 156.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 156.60 | 157.12 | 156.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 156.60 | 157.12 | 156.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 156.65 | 157.03 | 156.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 157.83 | 157.14 | 156.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 153.83 | 156.39 | 156.33 | SL hit (close<static) qty=1.00 sl=155.96 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 153.80 | 155.87 | 156.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 152.71 | 153.83 | 154.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 158.68 | 154.57 | 154.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 158.68 | 154.57 | 154.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 158.68 | 154.57 | 154.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 158.68 | 154.57 | 154.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 157.91 | 155.24 | 155.13 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 153.85 | 155.90 | 155.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 152.84 | 155.29 | 155.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 154.23 | 153.88 | 154.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 10:15:00 | 154.23 | 153.88 | 154.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 154.23 | 153.88 | 154.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 154.23 | 153.88 | 154.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 154.70 | 153.88 | 154.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 154.70 | 153.88 | 154.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 155.65 | 154.23 | 154.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 153.70 | 154.23 | 154.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 153.91 | 154.17 | 154.38 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 156.45 | 154.87 | 154.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 15:15:00 | 157.30 | 156.01 | 155.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 158.89 | 159.97 | 158.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 158.89 | 159.97 | 158.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 161.64 | 160.57 | 159.29 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 157.86 | 159.07 | 159.14 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 160.47 | 158.99 | 158.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 13:15:00 | 162.38 | 160.19 | 159.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 09:15:00 | 161.35 | 161.95 | 161.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 161.35 | 161.95 | 161.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 161.35 | 161.95 | 161.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:00:00 | 161.35 | 161.95 | 161.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 160.86 | 161.73 | 161.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 162.28 | 161.77 | 161.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 161.41 | 161.68 | 161.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:45:00 | 161.61 | 161.71 | 161.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 09:15:00 | 178.51 | 176.85 | 172.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 182.52 | 183.53 | 183.63 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 184.67 | 183.77 | 183.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 187.40 | 184.61 | 184.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 189.30 | 191.03 | 189.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 189.30 | 191.03 | 189.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 189.30 | 191.03 | 189.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 189.42 | 191.03 | 189.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 191.90 | 191.20 | 189.36 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 175.65 | 186.96 | 188.06 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 186.51 | 183.85 | 183.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 187.83 | 184.64 | 184.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 185.10 | 185.22 | 184.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:15:00 | 188.00 | 185.22 | 184.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 09:15:00 | 197.40 | 194.68 | 191.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-22 11:15:00 | 206.80 | 203.56 | 200.24 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 62 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 198.92 | 202.83 | 203.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 195.45 | 201.35 | 202.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 198.66 | 197.62 | 199.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 198.66 | 197.62 | 199.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 198.66 | 197.62 | 199.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 199.05 | 197.62 | 199.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 200.80 | 198.26 | 199.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 200.80 | 198.26 | 199.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 199.80 | 198.57 | 199.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 199.20 | 198.57 | 199.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 201.72 | 199.43 | 199.68 | SL hit (close>static) qty=1.00 sl=200.95 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 201.98 | 199.94 | 199.89 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 197.77 | 199.65 | 199.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 195.70 | 197.45 | 198.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 201.00 | 198.16 | 198.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 201.00 | 198.16 | 198.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 201.00 | 198.16 | 198.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 195.97 | 198.15 | 198.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 196.55 | 197.78 | 198.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 200.15 | 198.63 | 198.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 200.15 | 198.63 | 198.61 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 196.70 | 198.80 | 198.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 10:15:00 | 195.75 | 198.19 | 198.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 195.54 | 194.82 | 196.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:30:00 | 194.78 | 194.82 | 196.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 196.29 | 195.11 | 196.29 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 200.11 | 197.10 | 196.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 202.36 | 199.96 | 198.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 199.62 | 199.89 | 198.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 199.62 | 199.89 | 198.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 195.11 | 198.93 | 198.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 195.11 | 198.93 | 198.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 193.21 | 197.79 | 197.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 192.10 | 195.93 | 196.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 194.64 | 194.28 | 195.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 204.35 | 194.28 | 195.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 204.18 | 196.26 | 196.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 204.18 | 196.26 | 196.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 204.50 | 197.91 | 197.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 206.96 | 203.97 | 201.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 202.60 | 205.07 | 202.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 15:15:00 | 202.60 | 205.07 | 202.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 202.60 | 205.07 | 202.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 211.86 | 205.07 | 202.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 213.93 | 215.31 | 215.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 213.93 | 215.31 | 215.34 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 217.50 | 215.62 | 215.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 218.61 | 216.22 | 215.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 13:15:00 | 220.13 | 220.46 | 218.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:00:00 | 220.13 | 220.46 | 218.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 221.46 | 220.66 | 219.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 219.60 | 220.66 | 219.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 220.98 | 222.04 | 221.00 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 218.67 | 220.82 | 220.93 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 222.26 | 221.11 | 221.05 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 219.89 | 220.90 | 221.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 219.11 | 219.98 | 220.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 221.54 | 220.30 | 220.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 221.54 | 220.30 | 220.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 221.54 | 220.30 | 220.57 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 222.19 | 220.99 | 220.83 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 219.87 | 220.78 | 220.81 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 221.45 | 220.86 | 220.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 13:15:00 | 223.08 | 221.31 | 221.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 221.80 | 222.28 | 221.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:00:00 | 221.80 | 222.28 | 221.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 224.16 | 222.66 | 221.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 226.60 | 223.41 | 222.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:15:00 | 225.15 | 223.53 | 223.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:30:00 | 225.23 | 224.18 | 223.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 225.17 | 223.92 | 223.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 226.22 | 224.97 | 224.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 226.22 | 224.97 | 224.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 231.65 | 226.31 | 224.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 233.00 | 228.25 | 226.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-11 09:15:00 | 249.26 | 243.78 | 239.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 240.11 | 242.83 | 243.04 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 247.32 | 243.25 | 242.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 247.72 | 244.14 | 243.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 254.40 | 254.90 | 252.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:15:00 | 255.76 | 254.90 | 252.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 254.97 | 255.50 | 253.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 254.64 | 255.50 | 253.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 273.18 | 274.35 | 271.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 271.22 | 274.35 | 271.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 269.65 | 273.16 | 271.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 269.65 | 273.16 | 271.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 269.44 | 272.42 | 271.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 269.44 | 272.42 | 271.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 269.15 | 270.96 | 270.99 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 273.20 | 271.36 | 271.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 275.88 | 272.26 | 271.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 272.70 | 278.47 | 276.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 272.70 | 278.47 | 276.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 272.70 | 278.47 | 276.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 272.70 | 278.47 | 276.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 273.80 | 277.54 | 276.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 275.00 | 277.54 | 276.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 273.60 | 275.59 | 275.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 273.60 | 275.59 | 275.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 273.20 | 275.11 | 275.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 276.05 | 274.04 | 274.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 276.05 | 274.04 | 274.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 276.05 | 274.04 | 274.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 276.15 | 274.04 | 274.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 276.20 | 274.47 | 274.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 275.55 | 274.47 | 274.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 275.60 | 274.64 | 274.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 275.60 | 274.87 | 274.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 275.60 | 274.87 | 274.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 10:15:00 | 276.70 | 275.24 | 274.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 275.20 | 275.79 | 275.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 275.20 | 275.79 | 275.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 275.20 | 275.79 | 275.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 275.20 | 275.79 | 275.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 275.55 | 275.74 | 275.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 275.55 | 275.74 | 275.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 275.10 | 275.61 | 275.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:15:00 | 275.50 | 275.61 | 275.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 275.15 | 275.52 | 275.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:30:00 | 276.55 | 275.57 | 275.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 277.00 | 275.92 | 275.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 276.35 | 276.12 | 275.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 276.50 | 276.20 | 275.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 274.10 | 275.78 | 275.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 274.10 | 275.78 | 275.68 | SL hit (close<static) qty=1.00 sl=274.60 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 273.85 | 275.39 | 275.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 273.35 | 274.75 | 275.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 15:15:00 | 275.00 | 274.80 | 275.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 275.50 | 274.80 | 275.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 274.25 | 274.69 | 275.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 272.85 | 274.33 | 274.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 271.00 | 268.21 | 268.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 271.00 | 268.21 | 268.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 15:15:00 | 272.30 | 269.48 | 268.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 269.05 | 269.52 | 268.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 269.05 | 269.52 | 268.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 269.05 | 269.52 | 268.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 269.15 | 269.52 | 268.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 270.30 | 269.68 | 269.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 269.75 | 269.68 | 269.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 268.75 | 269.65 | 269.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 268.75 | 269.65 | 269.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 268.60 | 269.44 | 269.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 269.15 | 269.44 | 269.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 266.10 | 268.77 | 268.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 266.10 | 268.77 | 268.89 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 271.00 | 268.86 | 268.64 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 260.60 | 267.44 | 268.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 257.55 | 262.98 | 265.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 15:15:00 | 249.80 | 249.71 | 252.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 249.60 | 249.71 | 252.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 253.00 | 250.44 | 252.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 253.00 | 250.44 | 252.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 255.15 | 251.38 | 252.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 255.15 | 251.38 | 252.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 259.20 | 253.67 | 253.51 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 251.35 | 253.63 | 253.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 251.00 | 253.11 | 253.52 | Break + close below crossover candle low |

### Cycle 91 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 256.80 | 253.85 | 253.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 11:15:00 | 262.95 | 256.03 | 254.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 278.65 | 279.25 | 273.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:45:00 | 279.00 | 279.25 | 273.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 275.20 | 277.87 | 274.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 278.35 | 276.71 | 275.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 273.45 | 275.45 | 275.03 | SL hit (close<static) qty=1.00 sl=273.65 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 271.85 | 274.39 | 274.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 271.10 | 273.73 | 274.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 271.10 | 271.02 | 272.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 14:45:00 | 271.95 | 271.02 | 272.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 268.75 | 270.58 | 271.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 267.50 | 270.04 | 271.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 272.95 | 270.61 | 270.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 272.95 | 270.61 | 270.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 283.20 | 273.41 | 271.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 285.85 | 286.22 | 283.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 285.85 | 286.22 | 283.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 283.40 | 285.47 | 283.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 283.40 | 285.47 | 283.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 283.00 | 284.98 | 283.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 285.05 | 284.98 | 283.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 285.05 | 286.75 | 286.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 285.05 | 286.75 | 286.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 282.95 | 285.70 | 286.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 282.50 | 282.48 | 283.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:45:00 | 281.85 | 282.48 | 283.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 282.40 | 280.53 | 281.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 282.40 | 280.53 | 281.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 280.10 | 280.44 | 281.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 279.10 | 280.49 | 281.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 279.80 | 280.49 | 281.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 279.90 | 280.41 | 280.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 279.95 | 280.32 | 280.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 278.50 | 279.95 | 280.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 274.40 | 278.27 | 279.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 274.10 | 277.60 | 279.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:30:00 | 274.40 | 277.01 | 278.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 281.75 | 278.42 | 278.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 281.75 | 278.42 | 278.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 289.20 | 282.51 | 280.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 287.85 | 288.99 | 286.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 287.85 | 288.99 | 286.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 289.50 | 289.09 | 287.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 290.40 | 289.35 | 287.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 290.75 | 289.15 | 288.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 291.70 | 290.26 | 289.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 290.75 | 290.65 | 289.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 289.25 | 290.70 | 290.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 292.85 | 290.50 | 290.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 292.50 | 291.61 | 290.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 288.00 | 290.02 | 290.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 288.00 | 290.02 | 290.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 286.10 | 288.32 | 289.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 288.50 | 286.74 | 288.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 288.50 | 286.74 | 288.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 288.50 | 286.74 | 288.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 289.40 | 286.74 | 288.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 292.05 | 287.80 | 288.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 292.85 | 287.80 | 288.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 292.50 | 288.74 | 288.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 292.95 | 288.74 | 288.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 291.25 | 289.25 | 289.06 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 288.10 | 289.81 | 289.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 287.20 | 288.81 | 289.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 290.10 | 288.95 | 289.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 290.10 | 288.95 | 289.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 290.10 | 288.95 | 289.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 289.75 | 288.95 | 289.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 291.10 | 289.38 | 289.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 291.10 | 289.38 | 289.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 291.35 | 289.78 | 289.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 15:15:00 | 291.90 | 290.20 | 289.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 293.45 | 293.59 | 291.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 293.45 | 293.59 | 291.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 292.15 | 293.30 | 291.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 292.15 | 293.30 | 291.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 292.20 | 293.08 | 291.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 293.90 | 293.08 | 291.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 292.85 | 293.03 | 292.07 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 286.65 | 291.06 | 291.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 284.95 | 288.62 | 290.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 283.25 | 283.24 | 286.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 283.25 | 283.24 | 286.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 286.65 | 283.58 | 285.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 286.65 | 283.58 | 285.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 289.25 | 284.72 | 286.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 289.25 | 284.72 | 286.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 291.20 | 287.56 | 287.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 295.85 | 291.61 | 289.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 299.40 | 300.67 | 297.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 299.40 | 300.67 | 297.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 298.00 | 302.52 | 301.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 298.25 | 302.52 | 301.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 294.80 | 300.98 | 300.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 294.80 | 300.98 | 300.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 296.10 | 300.00 | 300.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 292.75 | 296.22 | 297.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 294.30 | 294.29 | 295.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 11:15:00 | 296.10 | 294.65 | 295.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 296.10 | 294.65 | 295.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 296.10 | 294.65 | 295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 294.95 | 294.71 | 295.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 295.90 | 294.71 | 295.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 298.10 | 295.39 | 295.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 298.10 | 295.39 | 295.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 298.90 | 296.09 | 296.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 298.90 | 296.09 | 296.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 300.00 | 296.87 | 296.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 301.25 | 297.75 | 297.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 297.70 | 297.74 | 297.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 297.70 | 297.74 | 297.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 297.70 | 297.74 | 297.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 297.70 | 297.74 | 297.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 295.50 | 297.29 | 296.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 295.50 | 297.29 | 296.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 296.65 | 297.16 | 296.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:00:00 | 297.15 | 297.16 | 296.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 299.40 | 300.40 | 300.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 299.40 | 300.40 | 300.50 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 302.40 | 300.75 | 300.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 304.10 | 301.42 | 300.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 303.80 | 303.98 | 302.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 303.80 | 303.98 | 302.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 303.80 | 303.98 | 302.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 303.80 | 303.98 | 302.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 305.95 | 307.54 | 305.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 305.95 | 307.54 | 305.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 305.70 | 307.17 | 305.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 304.70 | 307.17 | 305.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 306.00 | 306.94 | 305.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 305.95 | 306.94 | 305.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 305.35 | 306.62 | 305.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 305.40 | 306.62 | 305.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 306.45 | 306.59 | 305.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 307.95 | 306.51 | 305.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 327.25 | 328.33 | 328.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 327.25 | 328.33 | 328.47 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 333.55 | 329.31 | 328.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 338.40 | 332.14 | 330.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 12:15:00 | 333.60 | 334.82 | 332.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 333.60 | 334.82 | 332.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 333.30 | 334.49 | 333.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 334.10 | 334.49 | 333.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 333.85 | 334.36 | 333.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 329.70 | 334.36 | 333.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 324.30 | 332.35 | 332.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 324.30 | 332.35 | 332.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 326.00 | 331.08 | 331.72 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 334.55 | 331.01 | 330.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 337.00 | 333.73 | 332.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 331.60 | 334.04 | 333.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 331.60 | 334.04 | 333.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 331.60 | 334.04 | 333.22 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 332.65 | 333.19 | 333.21 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 333.50 | 333.26 | 333.24 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 331.55 | 332.91 | 333.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 11:15:00 | 329.60 | 331.28 | 332.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 328.20 | 327.90 | 329.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 328.20 | 327.90 | 329.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 332.50 | 328.13 | 329.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 332.50 | 328.13 | 329.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 332.85 | 329.07 | 329.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 332.45 | 329.07 | 329.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 332.75 | 329.81 | 329.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 341.20 | 333.11 | 331.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 356.30 | 356.67 | 353.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:00:00 | 356.30 | 356.67 | 353.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 354.60 | 356.02 | 354.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 354.75 | 356.02 | 354.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 355.90 | 355.99 | 354.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 354.75 | 355.99 | 354.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 351.85 | 355.23 | 354.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 351.85 | 355.23 | 354.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 349.45 | 354.07 | 354.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 348.20 | 350.82 | 352.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 355.35 | 350.89 | 351.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 355.35 | 350.89 | 351.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 355.35 | 350.89 | 351.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 355.35 | 350.89 | 351.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 356.70 | 352.05 | 351.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 10:15:00 | 362.45 | 357.41 | 355.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 361.30 | 363.43 | 360.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 12:00:00 | 361.30 | 363.43 | 360.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 359.45 | 362.15 | 360.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 359.45 | 362.15 | 360.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 358.00 | 361.32 | 360.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 361.10 | 361.32 | 360.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 360.60 | 360.66 | 360.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 360.25 | 360.83 | 360.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 355.85 | 359.54 | 359.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 355.85 | 359.54 | 359.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 14:15:00 | 354.70 | 358.57 | 359.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 359.30 | 357.92 | 358.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 359.30 | 357.92 | 358.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 359.30 | 357.92 | 358.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 359.30 | 357.92 | 358.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 360.25 | 358.39 | 359.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:15:00 | 360.05 | 358.39 | 359.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 359.60 | 358.63 | 359.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 359.60 | 358.63 | 359.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 362.45 | 359.39 | 359.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 363.40 | 360.20 | 359.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 357.40 | 360.39 | 360.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 357.40 | 360.39 | 360.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 357.40 | 360.39 | 360.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 357.40 | 360.39 | 360.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 354.50 | 359.21 | 359.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 353.10 | 356.62 | 358.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 348.30 | 348.14 | 350.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 348.20 | 348.14 | 350.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 347.90 | 345.96 | 347.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 348.50 | 345.96 | 347.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 348.55 | 346.47 | 347.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 348.55 | 346.47 | 347.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 345.85 | 346.85 | 347.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 345.45 | 346.46 | 347.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 345.00 | 346.54 | 347.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:30:00 | 345.35 | 346.37 | 346.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 344.65 | 346.46 | 346.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 346.80 | 346.53 | 346.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 348.75 | 346.53 | 346.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 353.80 | 347.98 | 347.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 353.80 | 347.98 | 347.41 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 345.15 | 347.80 | 347.85 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 348.70 | 347.85 | 347.81 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 345.95 | 347.45 | 347.64 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 349.40 | 347.70 | 347.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 350.65 | 348.44 | 347.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 15:15:00 | 348.05 | 348.56 | 348.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 348.05 | 348.56 | 348.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 348.05 | 348.56 | 348.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 353.90 | 348.56 | 348.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 357.55 | 361.37 | 361.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 357.55 | 361.37 | 361.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 351.20 | 355.24 | 357.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 354.85 | 354.66 | 356.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 355.65 | 354.66 | 356.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 355.30 | 354.99 | 356.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 352.75 | 354.40 | 355.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 352.15 | 353.57 | 355.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 352.60 | 352.57 | 353.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 358.40 | 355.05 | 354.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 358.40 | 355.05 | 354.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 358.95 | 355.83 | 355.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 358.70 | 360.21 | 358.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 358.70 | 360.21 | 358.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 358.70 | 360.21 | 358.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 358.70 | 360.21 | 358.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 360.80 | 360.33 | 358.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 357.50 | 360.33 | 358.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 360.55 | 360.72 | 359.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 358.05 | 360.72 | 359.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 360.15 | 360.61 | 359.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 361.55 | 360.61 | 359.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 360.65 | 360.62 | 359.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 360.65 | 360.62 | 359.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 354.95 | 359.48 | 359.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 354.95 | 359.48 | 359.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 355.25 | 358.64 | 358.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 351.80 | 356.33 | 357.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 353.70 | 350.34 | 353.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 353.70 | 350.34 | 353.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 353.70 | 350.34 | 353.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 355.95 | 350.34 | 353.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 349.20 | 350.11 | 352.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 347.95 | 350.09 | 352.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 356.75 | 353.19 | 353.33 | SL hit (close>static) qty=1.00 sl=355.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 357.65 | 354.08 | 353.72 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 349.70 | 352.99 | 353.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 348.10 | 352.01 | 352.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 351.05 | 350.86 | 352.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 351.05 | 350.86 | 352.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 351.05 | 350.86 | 352.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 351.35 | 350.86 | 352.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 349.25 | 348.47 | 350.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 351.65 | 348.47 | 350.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 347.10 | 348.20 | 349.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 344.85 | 347.76 | 349.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 344.25 | 347.11 | 348.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 12:15:00 | 327.61 | 337.64 | 340.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 12:15:00 | 327.04 | 337.64 | 340.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 326.45 | 325.88 | 331.84 | SL hit (close>ema200) qty=0.50 sl=325.88 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 343.95 | 335.22 | 334.78 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 337.15 | 340.91 | 341.13 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 348.30 | 342.16 | 341.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 353.45 | 345.20 | 343.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 349.25 | 350.43 | 347.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 349.25 | 350.43 | 347.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 349.25 | 350.43 | 347.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 352.35 | 350.82 | 348.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 344.85 | 350.83 | 349.36 | SL hit (close<static) qty=1.00 sl=345.95 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 343.80 | 348.08 | 348.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 337.35 | 342.65 | 344.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 338.90 | 338.56 | 341.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 338.90 | 338.56 | 341.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 342.60 | 339.37 | 341.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 342.60 | 339.37 | 341.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 344.05 | 340.31 | 341.53 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 344.85 | 342.47 | 342.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 345.35 | 343.05 | 342.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 345.35 | 347.79 | 346.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 345.35 | 347.79 | 346.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 345.35 | 347.79 | 346.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 345.35 | 347.79 | 346.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 346.60 | 347.55 | 346.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 344.90 | 347.55 | 346.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 346.55 | 347.35 | 346.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:15:00 | 348.00 | 347.35 | 346.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 348.30 | 347.54 | 346.74 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 341.80 | 345.86 | 346.14 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 347.15 | 345.63 | 345.55 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 344.15 | 345.46 | 345.51 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 346.05 | 345.57 | 345.56 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 342.20 | 345.05 | 345.33 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 347.25 | 345.51 | 345.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 349.05 | 346.22 | 345.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 353.90 | 353.91 | 351.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 14:30:00 | 355.60 | 354.38 | 352.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 345.50 | 352.54 | 351.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 345.50 | 352.54 | 351.86 | SL hit (close<ema400) qty=1.00 sl=351.86 alert=retest1 |

### Cycle 140 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 344.85 | 351.00 | 351.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 340.25 | 346.31 | 348.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 327.05 | 324.95 | 329.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 327.60 | 324.95 | 329.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 329.75 | 326.46 | 329.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 329.65 | 326.46 | 329.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 328.25 | 326.82 | 329.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 327.50 | 327.95 | 329.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 311.12 | 325.49 | 327.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 324.35 | 321.86 | 324.79 | SL hit (close>ema200) qty=0.50 sl=321.86 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 330.55 | 326.28 | 326.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 331.85 | 328.28 | 327.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 328.55 | 330.81 | 329.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 328.55 | 330.81 | 329.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 328.55 | 330.81 | 329.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 328.55 | 330.81 | 329.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 325.45 | 329.74 | 328.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 325.45 | 329.74 | 328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 323.25 | 328.44 | 328.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 323.25 | 328.44 | 328.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 322.50 | 327.25 | 327.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 317.50 | 325.30 | 326.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 311.75 | 311.71 | 315.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 311.75 | 311.71 | 315.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 314.55 | 312.28 | 315.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 316.20 | 312.28 | 315.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 319.15 | 314.06 | 315.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 319.15 | 314.06 | 315.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 321.60 | 315.57 | 316.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 323.50 | 315.57 | 316.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 318.50 | 316.40 | 316.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 318.50 | 316.40 | 316.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 318.50 | 316.82 | 316.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 325.00 | 319.11 | 317.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 317.85 | 323.52 | 321.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 317.85 | 323.52 | 321.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 317.85 | 323.52 | 321.34 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 311.40 | 318.58 | 319.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 310.55 | 313.39 | 315.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 302.80 | 299.16 | 303.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 302.80 | 299.16 | 303.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 302.80 | 299.16 | 303.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 304.25 | 299.16 | 303.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 303.15 | 299.96 | 303.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 302.55 | 299.96 | 303.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 301.30 | 300.23 | 303.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 302.85 | 300.23 | 303.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 303.20 | 300.82 | 303.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 313.70 | 300.82 | 303.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 313.30 | 303.32 | 304.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 313.30 | 303.32 | 304.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 314.75 | 305.60 | 305.14 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 305.85 | 307.05 | 307.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 296.45 | 304.37 | 305.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 300.10 | 297.54 | 300.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 300.10 | 297.54 | 300.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 300.10 | 297.54 | 300.73 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 302.80 | 302.11 | 302.07 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 289.65 | 299.75 | 301.02 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 303.20 | 299.53 | 299.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 306.95 | 301.02 | 300.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 307.00 | 307.44 | 305.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 327.00 | 307.44 | 305.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 332.40 | 338.18 | 334.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 332.40 | 338.18 | 334.11 | SL hit (close<ema400) qty=1.00 sl=334.11 alert=retest1 |

### Cycle 150 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 337.35 | 339.21 | 339.45 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 341.25 | 339.41 | 339.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 343.00 | 340.13 | 339.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 340.95 | 341.25 | 340.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 340.95 | 341.25 | 340.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 340.95 | 341.25 | 340.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 340.95 | 341.25 | 340.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 340.00 | 341.14 | 340.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 343.40 | 341.14 | 340.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 341.25 | 346.34 | 346.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 341.25 | 346.34 | 346.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 338.10 | 340.73 | 342.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 352.70 | 342.85 | 342.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 352.70 | 342.85 | 342.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 352.70 | 342.85 | 342.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 352.70 | 342.85 | 342.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 351.35 | 344.55 | 343.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 354.00 | 347.46 | 345.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 347.65 | 348.09 | 346.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 344.00 | 348.09 | 346.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 341.80 | 346.83 | 345.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 340.85 | 346.83 | 345.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 344.35 | 346.34 | 345.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 342.55 | 346.34 | 345.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 344.55 | 345.92 | 345.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 344.50 | 345.92 | 345.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 345.10 | 345.75 | 345.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:30:00 | 346.00 | 345.75 | 345.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 345.15 | 345.63 | 345.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 347.05 | 345.77 | 345.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 341.30 | 348.36 | 347.50 | SL hit (close<static) qty=1.00 sl=344.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 214.75 | 2024-05-13 12:15:00 | 217.95 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-28 09:15:00 | 228.10 | 2024-05-29 09:15:00 | 227.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-28 09:45:00 | 230.90 | 2024-05-29 09:15:00 | 227.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-28 12:45:00 | 228.25 | 2024-05-29 09:15:00 | 227.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-05-28 13:30:00 | 228.30 | 2024-05-29 09:15:00 | 227.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-06-26 15:15:00 | 235.15 | 2024-06-28 10:15:00 | 242.17 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-06-27 10:30:00 | 234.85 | 2024-06-28 10:15:00 | 242.17 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2024-07-11 10:15:00 | 227.95 | 2024-07-19 10:15:00 | 216.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:15:00 | 227.95 | 2024-07-22 13:15:00 | 217.60 | STOP_HIT | 0.50 | 4.54% |
| BUY | retest2 | 2024-08-09 09:30:00 | 211.35 | 2024-08-13 12:15:00 | 210.36 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-08-09 11:00:00 | 211.10 | 2024-08-13 12:15:00 | 210.36 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-08-09 11:45:00 | 212.50 | 2024-08-13 13:15:00 | 209.26 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-08-12 10:30:00 | 211.50 | 2024-08-13 13:15:00 | 209.26 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-08-12 14:00:00 | 213.98 | 2024-08-13 13:15:00 | 209.26 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-08-13 09:30:00 | 214.22 | 2024-08-13 13:15:00 | 209.26 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-08-30 10:45:00 | 221.63 | 2024-09-02 11:15:00 | 223.24 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-10 10:15:00 | 216.48 | 2024-09-10 12:15:00 | 218.88 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-11 13:00:00 | 216.65 | 2024-09-12 12:15:00 | 219.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-11 13:30:00 | 216.00 | 2024-09-12 12:15:00 | 219.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-09-12 09:45:00 | 215.95 | 2024-09-12 12:15:00 | 219.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-09-17 12:15:00 | 224.20 | 2024-10-03 13:15:00 | 234.30 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2024-09-19 11:45:00 | 224.21 | 2024-10-03 13:15:00 | 234.30 | STOP_HIT | 1.00 | 4.50% |
| SELL | retest2 | 2024-10-10 10:30:00 | 225.49 | 2024-10-15 09:15:00 | 225.01 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-10-10 11:00:00 | 225.64 | 2024-10-15 09:15:00 | 225.01 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-10-10 11:30:00 | 224.68 | 2024-10-15 09:15:00 | 225.01 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-10-21 12:00:00 | 217.34 | 2024-10-22 14:15:00 | 206.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 217.34 | 2024-10-23 10:15:00 | 211.63 | STOP_HIT | 0.50 | 2.63% |
| SELL | retest2 | 2024-10-21 14:00:00 | 216.43 | 2024-10-25 10:15:00 | 205.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 216.43 | 2024-10-29 11:15:00 | 203.13 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2024-11-04 09:15:00 | 201.65 | 2024-11-06 11:15:00 | 205.56 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-11-11 12:15:00 | 200.40 | 2024-11-13 09:15:00 | 190.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 200.77 | 2024-11-13 09:15:00 | 190.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 12:15:00 | 200.40 | 2024-11-14 12:15:00 | 190.11 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2024-11-11 13:15:00 | 200.77 | 2024-11-14 12:15:00 | 190.11 | STOP_HIT | 0.50 | 5.31% |
| BUY | retest2 | 2024-11-29 13:15:00 | 194.23 | 2024-12-09 13:15:00 | 197.16 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-11-29 13:45:00 | 194.00 | 2024-12-09 13:15:00 | 197.16 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2024-12-02 10:15:00 | 194.30 | 2024-12-09 13:15:00 | 197.16 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-12-02 11:15:00 | 194.17 | 2024-12-09 13:15:00 | 197.16 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2024-12-03 09:15:00 | 195.80 | 2024-12-09 13:15:00 | 197.16 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-12-16 11:30:00 | 195.90 | 2024-12-19 09:15:00 | 186.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:30:00 | 195.90 | 2024-12-19 14:15:00 | 190.72 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-12-17 09:30:00 | 195.68 | 2024-12-20 15:15:00 | 185.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 195.68 | 2024-12-24 09:15:00 | 187.57 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-01-08 09:15:00 | 176.50 | 2025-01-10 09:15:00 | 169.68 | PARTIAL | 0.50 | 3.86% |
| SELL | retest2 | 2025-01-08 10:15:00 | 178.61 | 2025-01-10 09:15:00 | 169.83 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-01-08 15:15:00 | 178.77 | 2025-01-10 14:15:00 | 167.67 | PARTIAL | 0.50 | 6.21% |
| SELL | retest2 | 2025-01-08 09:15:00 | 176.50 | 2025-01-13 09:15:00 | 172.48 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2025-01-08 10:15:00 | 178.61 | 2025-01-13 09:15:00 | 172.48 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-01-08 15:15:00 | 178.77 | 2025-01-13 09:15:00 | 172.48 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2025-01-16 09:15:00 | 177.43 | 2025-01-21 14:15:00 | 174.63 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-23 11:15:00 | 173.38 | 2025-01-27 10:15:00 | 164.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 174.08 | 2025-01-27 10:15:00 | 165.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 173.90 | 2025-01-27 10:15:00 | 165.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 173.20 | 2025-01-27 10:15:00 | 164.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 172.08 | 2025-01-28 09:15:00 | 163.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 173.38 | 2025-01-28 11:15:00 | 167.08 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-01-23 14:15:00 | 174.08 | 2025-01-28 11:15:00 | 167.08 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-01-23 15:15:00 | 173.90 | 2025-01-28 11:15:00 | 167.08 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-01-24 09:30:00 | 173.20 | 2025-01-28 11:15:00 | 167.08 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-01-24 13:30:00 | 172.08 | 2025-01-28 11:15:00 | 167.08 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-02-10 09:30:00 | 167.76 | 2025-02-12 09:15:00 | 159.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:45:00 | 168.10 | 2025-02-12 09:15:00 | 159.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 167.76 | 2025-02-12 15:15:00 | 159.75 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-02-10 11:45:00 | 168.10 | 2025-02-12 15:15:00 | 159.75 | STOP_HIT | 0.50 | 4.97% |
| BUY | retest2 | 2025-02-21 15:00:00 | 157.83 | 2025-02-24 09:15:00 | 153.83 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-03-17 11:30:00 | 162.28 | 2025-03-21 09:15:00 | 178.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:30:00 | 161.41 | 2025-03-21 09:15:00 | 177.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 14:45:00 | 161.61 | 2025-03-21 09:15:00 | 177.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-04-15 09:15:00 | 188.00 | 2025-04-17 09:15:00 | 197.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-15 09:15:00 | 188.00 | 2025-04-22 11:15:00 | 206.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-23 11:15:00 | 204.09 | 2025-04-25 09:15:00 | 198.92 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-04-24 13:00:00 | 204.10 | 2025-04-25 09:15:00 | 198.92 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-04-24 13:45:00 | 203.93 | 2025-04-25 09:15:00 | 198.92 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-04-28 14:15:00 | 199.20 | 2025-04-29 09:15:00 | 201.72 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-05-02 12:15:00 | 195.97 | 2025-05-05 10:15:00 | 200.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-05-02 14:15:00 | 196.55 | 2025-05-05 10:15:00 | 200.15 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-14 09:15:00 | 211.86 | 2025-05-20 14:15:00 | 213.93 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-06-03 09:15:00 | 226.60 | 2025-06-11 09:15:00 | 249.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 12:15:00 | 225.15 | 2025-06-11 09:15:00 | 247.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 14:30:00 | 225.23 | 2025-06-11 09:15:00 | 247.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 12:15:00 | 225.17 | 2025-06-11 09:15:00 | 247.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 12:30:00 | 233.00 | 2025-06-13 10:15:00 | 240.11 | STOP_HIT | 1.00 | 3.05% |
| BUY | retest2 | 2025-07-02 11:15:00 | 275.00 | 2025-07-02 15:15:00 | 273.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-07-04 11:15:00 | 275.55 | 2025-07-07 09:15:00 | 275.60 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-07-04 14:00:00 | 275.60 | 2025-07-07 09:15:00 | 275.60 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-07-08 13:30:00 | 276.55 | 2025-07-09 11:15:00 | 274.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-08 14:45:00 | 277.00 | 2025-07-09 11:15:00 | 274.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-09 10:15:00 | 276.35 | 2025-07-09 11:15:00 | 274.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-09 11:00:00 | 276.50 | 2025-07-09 11:15:00 | 274.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-10 10:30:00 | 272.85 | 2025-07-21 13:15:00 | 271.00 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-07-23 09:15:00 | 269.15 | 2025-07-23 09:15:00 | 266.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-07 15:00:00 | 278.35 | 2025-08-08 12:15:00 | 273.45 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-12 15:00:00 | 267.50 | 2025-08-14 14:15:00 | 272.95 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-08-21 09:15:00 | 285.05 | 2025-08-25 14:15:00 | 285.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-08-29 13:45:00 | 279.10 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-08-29 14:15:00 | 279.80 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-01 14:15:00 | 279.90 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-01 15:00:00 | 279.95 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-02 13:15:00 | 274.40 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-09-02 13:45:00 | 274.10 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-09-02 14:30:00 | 274.40 | 2025-09-05 09:15:00 | 281.75 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-09-10 12:00:00 | 290.40 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-11 09:45:00 | 290.75 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-09-12 09:15:00 | 291.70 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-09-12 13:00:00 | 290.75 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-09-16 09:15:00 | 292.85 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-16 12:15:00 | 292.50 | 2025-09-17 09:15:00 | 288.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-10-14 14:00:00 | 297.15 | 2025-10-17 13:15:00 | 299.40 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-10-27 09:15:00 | 307.95 | 2025-11-06 14:15:00 | 327.25 | STOP_HIT | 1.00 | 6.27% |
| BUY | retest2 | 2025-12-11 09:15:00 | 361.10 | 2025-12-11 13:15:00 | 355.85 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-11 11:15:00 | 360.60 | 2025-12-11 13:15:00 | 355.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-11 11:45:00 | 360.25 | 2025-12-11 13:15:00 | 355.85 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-22 14:30:00 | 345.45 | 2025-12-24 09:15:00 | 353.80 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-12-23 09:30:00 | 345.00 | 2025-12-24 09:15:00 | 353.80 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-12-23 11:30:00 | 345.35 | 2025-12-24 09:15:00 | 353.80 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-23 14:30:00 | 344.65 | 2025-12-24 09:15:00 | 353.80 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-12-31 09:15:00 | 353.90 | 2026-01-07 09:15:00 | 357.55 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2026-01-13 11:45:00 | 352.75 | 2026-01-14 14:15:00 | 358.40 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-01-13 12:30:00 | 352.15 | 2026-01-14 14:15:00 | 358.40 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-14 11:30:00 | 352.60 | 2026-01-14 14:15:00 | 358.40 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-01-22 11:30:00 | 347.95 | 2026-01-23 09:15:00 | 356.75 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-01-28 11:30:00 | 344.85 | 2026-02-01 12:15:00 | 327.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 10:30:00 | 344.25 | 2026-02-01 12:15:00 | 327.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 11:30:00 | 344.85 | 2026-02-02 13:15:00 | 326.45 | STOP_HIT | 0.50 | 5.34% |
| SELL | retest2 | 2026-01-29 10:30:00 | 344.25 | 2026-02-02 13:15:00 | 326.45 | STOP_HIT | 0.50 | 5.17% |
| BUY | retest2 | 2026-02-10 12:00:00 | 352.35 | 2026-02-11 09:15:00 | 344.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2026-02-26 14:30:00 | 355.60 | 2026-02-27 09:15:00 | 345.50 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-06 14:30:00 | 327.50 | 2026-03-09 09:15:00 | 311.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 327.50 | 2026-03-09 14:15:00 | 324.35 | STOP_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2026-04-08 09:15:00 | 327.00 | 2026-04-13 09:15:00 | 332.40 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2026-04-13 10:45:00 | 334.60 | 2026-04-17 10:15:00 | 337.35 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2026-04-21 09:15:00 | 343.40 | 2026-04-24 11:15:00 | 341.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-05-04 09:15:00 | 347.05 | 2026-05-04 14:15:00 | 341.30 | STOP_HIT | 1.00 | -1.66% |
